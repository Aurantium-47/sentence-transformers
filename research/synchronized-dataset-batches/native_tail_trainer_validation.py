"""Additional nonzero-tail/unequal-frequency native-objective validation.

Derived test-only from the submitted v1 harness; production and loss sources are unchanged.

The worker uses the real Trainer loop, Accelerate, loss classes, checkpoint save
and load, and Gloo DDP. Only observation and early stopping are subclassed.
The single-process oracle runs in the parent after all worker groups have exited.
Pass TrainingArguments kwargs with --options-json. Use the normal editable
install described in README.md; no runtime import overlay is needed.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


FIXTURE = {"A_rows": 65, "B_rows": 9, "vocab_size": 148, "embedding_dimension": 8,
           "batch_sizes": {"A": 8, "B": 2}, "learning_rate": 0.03,
           "momentum": 0.9, "model_seed": 123, "data_seed": 731,
           "loss_scale": 1.0, "mini_batch_size": 2, "world_size": 2}


def model_and_data():
    import torch
    from datasets import Dataset, DatasetDict
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules as models

    torch.manual_seed(FIXTURE["model_seed"])
    model = SentenceTransformer(modules=[
        models.BoW([f"word{i}" for i in range(FIXTURE["vocab_size"])]),
        models.Dense(FIXTURE["vocab_size"], FIXTURE["embedding_dimension"],
                     bias=False, activation_function=torch.nn.Identity()),
    ], device="cpu")
    a_rows, b_rows = FIXTURE["A_rows"], FIXTURE["B_rows"]
    a = Dataset.from_dict({"anchor": [f"word{i}" for i in range(a_rows)],
                           "positive": [f"word{i + a_rows}" for i in range(a_rows)]})
    b = Dataset.from_dict({"anchor": [f"word{i + 2*a_rows}" for i in range(b_rows)],
                           "positive": [f"word{i + 2*a_rows + b_rows}" for i in range(b_rows)],
                           "label": [float(i % 4) for i in range(b_rows)]})
    return model, DatasetDict({"A": a, "B": b})


def make_losses(model, gather):
    from sentence_transformers.sentence_transformer import losses
    return {
        "A": losses.CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=FIXTURE["mini_batch_size"],
            scale=FIXTURE["loss_scale"], gather_across_devices=gather),
        "B": losses.CoSENTLoss(model, scale=FIXTURE["loss_scale"]),
    }


def parameter_values(model, gradient=False):
    result = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradient else parameter
        result[name.removeprefix("module.")] = None if value is None else value.detach().cpu().flatten().tolist()
    return result


def momentum_values(model, optimizer):
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    result = {}
    for name, parameter in model.named_parameters():
        value = optimizer.state.get(parameter, {}).get("momentum_buffer")
        result[name.removeprefix("module.")] = None if value is None else value.detach().cpu().flatten().tolist()
    return result


def worker(args):
    import hashlib
    import importlib.metadata
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.base.sampler import MultiDatasetBatchSamplers

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    rank = int(os.environ["RANK"])
    run = args.output_dir / f"g{args.g}_{args.phase}"
    run.mkdir(parents=True, exist_ok=True)
    trace = {"rank": rank, "phase": args.phase, "G": args.g, "fixture": FIXTURE,
             "microsteps": [], "updates": [], "epochs": [], "status": "started"}
    trace["versions"] = {p: importlib.metadata.version(p) for p in
                         ["torch", "transformers", "datasets", "sentence-transformers", "accelerate"]}
    model, data = model_and_data()
    trace["initial_parameters"] = parameter_values(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=FIXTURE["learning_rate"], momentum=FIXTURE["momentum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    options = json.loads(args.options_json.read_text())
    strategy = options.get("multi_dataset_batch_sampler", "round_robin")
    capacities = [FIXTURE["A_rows"] // 16, FIXTURE["B_rows"] // 4]
    retained_counts = [min(capacities)] * 2 if strategy == "round_robin" else capacities
    microsteps_per_epoch = sum(retained_counts)
    assert microsteps_per_epoch % args.g == 0
    total_updates = 2 * microsteps_per_epoch // args.g
    trace.update(strategy=strategy, retained_counts=retained_counts,
                 microsteps_per_epoch=microsteps_per_epoch)
    stop_at = 3 if args.g == 1 else 1

    class Observe(TrainerCallback):
        def on_epoch_begin(self, arguments, state, control, **kwargs):
            trace["epochs"].append({"state_epoch": state.epoch, "global_step": state.global_step})

        def on_pre_optimizer_step(self, arguments, state, control, model, optimizer, **kwargs):
            trace["updates"].append({"step": state.global_step + 1,
                                     "parameters_before": parameter_values(model),
                                     "gradients": parameter_values(model, gradient=True),
                                     "momentum_before": momentum_values(model, optimizer)})

        def on_optimizer_step(self, arguments, state, control, model, optimizer, **kwargs):
            trace["updates"][-1].update(parameters_after=parameter_values(model),
                                       momentum_after=momentum_values(model, optimizer))

        def on_step_end(self, arguments, state, control, **kwargs):
            if args.phase == "prefix" and state.global_step == stop_at:
                control.should_training_stop = True
                control.should_save = True
            return control

    class ObserveTrainer(SentenceTransformerTrainer):
        def compute_loss(self, model, inputs, **kwargs):
            # Observe before a loss mutates the feature dictionaries. The one-hot
            # features identify real sampled rows independently of sampler metadata.
            task = inputs.get("dataset_name")
            anchor = inputs["anchor_sentence_embedding"]
            positive = inputs["positive_sentence_embedding"]
            record = {"update": self.state.global_step + 1,
                      "state_epoch": self.state.epoch,
                      "task": task,
                      "anchor_ids": anchor.argmax(dim=1).tolist(),
                      "positive_ids": positive.argmax(dim=1).tolist(),
                      "labels": inputs["label"].tolist() if "label" in inputs else None}
            assert bool((anchor.sum(dim=1) == 1).all())
            assert bool((positive.sum(dim=1) == 1).all())
            value = super().compute_loss(model, inputs, **kwargs)
            assert not isinstance(value, tuple)
            record["loss"] = float(value.detach())
            trace["microsteps"].append(record)
            print(json.dumps({"rank": rank, "phase": args.phase, "G": args.g,
                              "microstep": len(trace["microsteps"]), "task": task,
                              "update": record["update"], "loss": record["loss"]}), flush=True)
            return value

    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run / "checkpoints"), use_cpu=True,
        per_device_train_batch_size=8,
        multi_dataset_batch_sampler=options.pop("multi_dataset_batch_sampler", MultiDatasetBatchSamplers.ROUND_ROBIN),
        dataloader_drop_last=True, dataloader_num_workers=0,
        dataloader_pin_memory=False, max_steps=total_updates,
        gradient_accumulation_steps=args.g, seed=FIXTURE["data_seed"],
        data_seed=FIXTURE["data_seed"], learning_rate=FIXTURE["learning_rate"],
        save_strategy="steps" if args.phase == "prefix" else "no",
        save_steps=stop_at, logging_strategy="no", report_to=[], disable_tqdm=True,
        max_grad_norm=0.0, ddp_backend="gloo", ddp_timeout=15,
        ddp_find_unused_parameters=False,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
        **options,
    )
    trainer = ObserveTrainer(model=model, args=arguments, train_dataset=data,
                             loss=make_losses(model, gather=True), callbacks=[Observe()],
                             optimizers=(optimizer, scheduler))
    try:
        loader = trainer.get_train_dataloader()
        trace["prepared_loader"] = {"class": type(loader).__name__, "length": len(loader),
                                    "total_batch_size": loader.total_batch_size}
        trace["device_configuration"] = {"arguments_device": str(arguments.device),
                                         "ACCELERATE_TORCH_DEVICE": os.environ.get("ACCELERATE_TORCH_DEVICE")}
        trace["imported_source"] = {
            "trainer": sys.modules[SentenceTransformerTrainer.__module__].__file__,
            "base_trainer": sys.modules["sentence_transformers.base.trainer"].__file__,
        }
        trace["source_hashes"] = {}
        for module_name in ["sentence_transformers.base.trainer", "sentence_transformers.base.sampler",
                            "sentence_transformers.base.training_args", "accelerate.data_loader", "accelerate.accelerator"]:
            path = Path(sys.modules[module_name].__file__)
            trace["source_hashes"][module_name] = {"path": str(path),
                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        resume = None
        if args.phase == "resume":
            resume = str(args.output_dir / f"g{args.g}_prefix" / "checkpoints" / f"checkpoint-{stop_at}")
        outcome = trainer.train(resume_from_checkpoint=resume)
        trace["global_step"] = trainer.state.global_step
        trace["state_epoch"] = trainer.state.epoch
        trace["final_parameters"] = parameter_values(model)
        trace["final_momentum"] = momentum_values(model, trainer.optimizer)
        trace["trainer_metrics"] = outcome.metrics
        assert "train_samples_per_second" not in outcome.metrics
        assert all("train_samples_per_second" not in item for item in trainer.state.log_history)
        assert "train_runtime" in outcome.metrics and "train_steps_per_second" in outcome.metrics
        trace["status"] = "completed"
    except BaseException as exc:
        trace.update(status="exception", exception_type=type(exc).__name__, error=str(exc),
                     traceback=traceback.format_exc())
        raise
    finally:
        (run / f"rank{rank}.json").write_text(json.dumps(trace, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


def run_bounded(cmd, env, log_path, timeout=90):
    import psutil
    with log_path.open("w") as logfile:
        process = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        killed = []
        try:
            code = process.wait(timeout=timeout)
            return {"returncode": code, "timed_out": False, "killed_pids": []}
        except subprocess.TimeoutExpired:
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            for child in reversed(children):
                try:
                    child.kill()
                    killed.append(child.pid)
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait(timeout=5)
            return {"returncode": process.returncode, "timed_out": True, "killed_pids": killed}


def assert_close_dict(actual, expected, label, summary, atol=2e-6, rtol=2e-5):
    import torch
    assert actual.keys() == expected.keys(), (label, actual.keys(), expected.keys())
    max_error = 0.0
    for name in actual:
        a, b = actual[name], expected[name]
        assert (a is None) == (b is None), (label, name, a is None, b is None)
        if a is not None:
            ta, tb = torch.tensor(a), torch.tensor(b)
            max_error = max(max_error, float((ta - tb).abs().max()))
            torch.testing.assert_close(ta, tb, atol=atol, rtol=rtol, msg=f"{label}: {name}")
    summary[label] = max_error


def verify_group(output_dir, g):
    import torch
    assert not torch.distributed.is_initialized()
    traces = {}
    for phase in ["full", "prefix", "resume"]:
        traces[phase] = [json.loads((output_dir / f"g{g}_{phase}" / f"rank{rank}.json").read_text())
                         for rank in range(2)]
        assert all(t["status"] == "completed" for t in traces[phase]), phase
    full0, full1 = traces["full"]
    strategy = full0["strategy"]
    assert full1["strategy"] == strategy
    capacities = [FIXTURE["A_rows"] // 16, FIXTURE["B_rows"] // 4]
    counts = [min(capacities)] * 2 if strategy == "round_robin" else capacities
    microsteps_per_epoch = sum(counts)
    total_microsteps = 2 * microsteps_per_epoch
    assert full0["global_step"] == full1["global_step"] == total_microsteps // g
    assert len(full0["microsteps"]) == len(full1["microsteps"]) == total_microsteps
    assert full0["prepared_loader"]["length"] == full1["prepared_loader"]["length"] == microsteps_per_epoch
    assert full0["state_epoch"] == full1["state_epoch"] == 2.0
    assert [m["task"] for m in full0["microsteps"]] == [m["task"] for m in full1["microsteps"]]
    # Both strategies must align ranks. Coverage and native-loss oracle below
    # validate actual observed tasks rather than assuming round-robin ordering.
    summary = {"G": g, "updates": [], "max_errors": {}, "sample_coverage": [],
               "task_sequence": [m["task"] for m in full0["microsteps"]]}
    valid_rows = {"A": set(range(FIXTURE["A_rows"])),
                  "B": set(range(2*FIXTURE["A_rows"], 2*FIXTURE["A_rows"] + FIXTURE["B_rows"]))}
    retained_rows = {"A": counts[0] * 16, "B": counts[1] * 4}
    summary.update(strategy=strategy, retained_counts=counts, microsteps_per_epoch=microsteps_per_epoch)
    for epoch in range(2):
        observed = {"A": [], "B": []}
        task_counts = {"A": 0, "B": 0}
        for i in range(epoch * microsteps_per_epoch, (epoch + 1) * microsteps_per_epoch):
            m0, m1 = full0["microsteps"][i], full1["microsteps"][i]
            assert m0["task"] == m1["task"]
            task = m0["task"]
            task_counts[task] += 1
            for m in [m0, m1]:
                assert len(m["anchor_ids"]) == FIXTURE["batch_sizes"][task]
                assert len(m["positive_ids"]) == len(m["anchor_ids"])
            assert not set(m0["anchor_ids"]) & set(m1["anchor_ids"])
            for m in [m0, m1]:
                observed[task].extend(m["anchor_ids"])
                offset = FIXTURE["A_rows"] if task == "A" else FIXTURE["B_rows"]
                assert m["positive_ids"] == [x + offset for x in m["anchor_ids"]]
                if task == "B":
                    assert m["labels"] == [float((x - 2*FIXTURE["A_rows"]) % 4) for x in m["anchor_ids"]]
        assert task_counts == dict(zip(("A", "B"), counts))
        dropped = {}
        for task in ("A", "B"):
            assert len(observed[task]) == len(set(observed[task])) == retained_rows[task]
            assert set(observed[task]) <= valid_rows[task]
            dropped[task] = sorted(valid_rows[task] - set(observed[task]))
            assert len(dropped[task]) == len(valid_rows[task]) - retained_rows[task]
        if strategy == "round_robin":
            assert [m["task"] for m in full0["microsteps"][epoch*microsteps_per_epoch:(epoch+1)*microsteps_per_epoch]] == ["A", "B"] * counts[0]
        summary["sample_coverage"].append({"epoch": epoch, "A": observed["A"], "B": observed["B"],
                                           "dropped_anchor_ids": dropped, "task_counts": task_counts})
    assert summary["sample_coverage"][0]["A"] != summary["sample_coverage"][1]["A"]

    oracle, _ = model_and_data()
    criteria = make_losses(oracle, gather=False)
    optimizer = torch.optim.SGD(oracle.parameters(), lr=FIXTURE["learning_rate"], momentum=FIXTURE["momentum"])
    assert_close_dict(parameter_values(oracle), full0["initial_parameters"], "initial", summary["max_errors"], 0, 0)

    def features(m):
        fs = []
        for key in ["anchor_ids", "positive_ids"]:
            onehot = torch.zeros(len(m[key]), FIXTURE["vocab_size"])
            onehot[torch.arange(len(m[key])), torch.tensor(m[key])] = 1.0
            fs.append({"sentence_embedding": onehot})
        labels = None if m["labels"] is None else torch.tensor(m["labels"])
        return fs, labels

    for update in range(total_microsteps // g):
        optimizer.zero_grad(set_to_none=True)
        expected_losses = []
        for i in range(update * g, (update + 1) * g):
            m0, m1 = full0["microsteps"][i], full1["microsteps"][i]
            task = m0["task"]
            if task == "A":
                combined = {key: m0[key] + m1[key] for key in ["anchor_ids", "positive_ids"]}
                combined["labels"] = None
                loss = criteria[task](*features(combined))
            else:
                loss = sum(criteria[task](*features(m)) for m in [m0, m1]) / 2
            rank_mean = (m0["loss"] + m1["loss"]) / 2
            torch.testing.assert_close(torch.tensor(rank_mean), loss.detach(), atol=2e-6, rtol=2e-5)
            expected_losses.append(float(loss.detach()))
            (loss / g).backward()
        actual0, actual1 = full0["updates"][update], full1["updates"][update]
        for key in ["parameters_before", "gradients", "momentum_before", "parameters_after", "momentum_after"]:
            assert_close_dict(actual0[key], actual1[key], f"rank_agreement_update{update+1}_{key}", summary["max_errors"], 1e-7, 1e-6)
        assert_close_dict(actual0["parameters_before"], parameter_values(oracle), f"before_update{update+1}", summary["max_errors"])
        assert_close_dict(actual0["gradients"], parameter_values(oracle, True), f"gradient_update{update+1}", summary["max_errors"])
        assert_close_dict(actual0["momentum_before"], momentum_values(oracle, optimizer), f"momentum_before_update{update+1}", summary["max_errors"])
        optimizer.step()
        assert_close_dict(actual0["parameters_after"], parameter_values(oracle), f"after_update{update+1}", summary["max_errors"])
        assert_close_dict(actual0["momentum_after"], momentum_values(oracle, optimizer), f"momentum_after_update{update+1}", summary["max_errors"])
        summary["updates"].append({"update": update + 1, "tasks": [m["task"] for m in full0["microsteps"][update*g:(update+1)*g]],
                                   "oracle_microstep_losses": expected_losses, "window_loss_mean": sum(expected_losses)/g})
    assert any(abs(m["loss"]) > 1e-4 for m in full0["microsteps"] + full1["microsteps"] if m["task"] == "B")

    stop_at = 3 if g == 1 else 1
    for rank in range(2):
        prefix, resume, full = traces["prefix"][rank], traces["resume"][rank], traces["full"][rank]
        assert prefix["global_step"] == stop_at
        assert resume["global_step"] == full["global_step"]
        joined = prefix["microsteps"] + resume["microsteps"]
        assert len(joined) == len(full["microsteps"])
        for i, (a, b) in enumerate(zip(joined, full["microsteps"])):
            for key in ["update", "task", "anchor_ids", "positive_ids", "labels"]:
                assert a[key] == b[key], ("resume_sequence", rank, i, key, a[key], b[key])
            assert math.isclose(a["loss"], b["loss"], rel_tol=2e-5, abs_tol=2e-6)
        joined_updates = prefix["updates"] + resume["updates"]
        assert len(joined_updates) == len(full["updates"])
        for i, (a, b) in enumerate(zip(joined_updates, full["updates"])):
            assert a["step"] == b["step"]
            for key in ["parameters_before", "gradients", "momentum_before", "parameters_after", "momentum_after"]:
                assert_close_dict(a[key], b[key], f"resume_rank{rank}_update{i+1}_{key}", summary["max_errors"], 1e-7, 1e-6)
    summary["native_oracle_pass"] = True
    summary["second_epoch_pass"] = True
    summary["checkpoint_resume_pass"] = True
    summary["largest_absolute_error"] = max(summary["max_errors"].values())
    return summary


def parent(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TORCH_DISTRIBUTED_DEBUG="INFO", ACCELERATE_TORCH_DEVICE="cpu")
    runs = []
    groups = [args.g] if args.g else [1, 2]
    for g in groups:
        for phase in ["full", "prefix", "resume"]:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   "--nproc_per_node=2", str(Path(__file__).resolve()), "--worker", "--g", str(g),
                   "--phase", phase, "--output-dir", str(args.output_dir.resolve()),
                   "--options-json", str(args.options_json.resolve())]
            status = run_bounded(cmd, env, args.output_dir / f"g{g}_{phase}.log")
            status.update(G=g, phase=phase)
            runs.append(status)
            print(json.dumps(status), flush=True)
            (args.output_dir / "process_runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            if status["returncode"] != 0:
                raise RuntimeError(f"Worker failure: {status}; see log")
    report = {"scope": "real full Trainer plus native DDP objectives; no concatenated CoSENT claim", "runs": runs}
    report["groups"] = [verify_group(args.output_dir, g) for g in groups]
    report["passed"] = all(r["native_oracle_pass"] and r["second_epoch_pass"] and r["checkpoint_resume_pass"] for r in report["groups"])
    (args.output_dir / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "largest_errors": [g["largest_absolute_error"] for g in report["groups"]]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--g", type=int, choices=[1, 2])
    parser.add_argument("--phase", choices=["full", "prefix", "resume"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--options-json", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
