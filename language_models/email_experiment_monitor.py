#!/usr/bin/env python3
"""Periodic email monitor for language-model rollout experiments.

Sends periodic summaries and attaches generated plots.
SMTP credentials are loaded from environment and/or a JSON config file.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import smtplib
import ssl
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class SmtpSettings:
    host: str
    port: int
    username: str
    password: str
    from_email: str
    to_email: str
    use_starttls: bool


@dataclass
class DirectMxSettings:
    from_email: str
    to_email: str
    mx_host: str
    mx_port: int
    use_starttls: bool
    helo_hostname: str


DeliverySettings = Union[SmtpSettings, DirectMxSettings]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor experiment progress and email periodic summaries.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="language_models/results/token_stability",
        help="Directory containing per-seed experiment outputs.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots/language_models/token_stability",
        help="Directory containing generated plot files.",
    )
    parser.add_argument(
        "--email-config",
        type=str,
        default="language_models/email_monitor_config.json",
        help="JSON file with SMTP/email settings (optional, env vars can be used instead).",
    )
    parser.add_argument(
        "--from-email",
        type=str,
        default="",
        help="Override sender email address.",
    )
    parser.add_argument(
        "--to-email",
        type=str,
        default="",
        help="Override destination email address.",
    )
    parser.add_argument(
        "--experiment-pid",
        type=int,
        default=0,
        help="Optional experiment PID to monitor for running/finished state.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=3600,
        help="Email interval in seconds.",
    )
    parser.add_argument(
        "--subject-prefix",
        type=str,
        default="[ICR LM Monitor]",
        help="Prefix for outgoing email subject.",
    )
    parser.add_argument(
        "--max-attachment-mb",
        type=float,
        default=20.0,
        help="Max total attachment size target in MB before switching to zip fallback.",
    )
    parser.add_argument(
        "--no-attach-plots",
        action="store_true",
        help="Disable plot attachments (summary text only).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send a single summary and exit.",
    )
    parser.add_argument(
        "--stop-when-finished",
        action="store_true",
        help="Exit after sending a summary when experiment is no longer running.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    sval = str(value).strip().lower()
    if sval in {"1", "true", "yes", "y", "on"}:
        return True
    if sval in {"0", "false", "no", "n", "off"}:
        return False
    return default


def infer_direct_mx_host(to_email: str) -> str:
    domain = to_email.rsplit("@", 1)[-1].lower() if "@" in to_email else ""
    if domain in {"gmail.com", "googlemail.com"}:
        return "gmail-smtp-in.l.google.com"
    return ""


def load_delivery_settings(
    config_path: Path,
    from_email_override: str,
    to_email_override: str,
) -> Tuple[Optional[DeliverySettings], List[str]]:
    config = read_json(config_path) or {}

    def resolve(key_env: str, key_cfg: str, default: Any = "") -> Any:
        env_val = os.environ.get(key_env)
        if env_val not in {None, ""}:
            return env_val
        cfg_val = config.get(key_cfg)
        if cfg_val not in {None, ""}:
            return cfg_val
        return default

    delivery_mode = str(resolve("EMAIL_DELIVERY_MODE", "delivery_mode", "smtp_auth")).strip().lower()
    from_email = str(from_email_override or resolve("SMTP_FROM_EMAIL", "from_email", "")).strip()
    to_email = str(to_email_override or resolve("SMTP_TO_EMAIL", "to_email", "")).strip()

    if delivery_mode in {"direct", "direct_mx", "mx"}:
        mx_host = str(resolve("DIRECT_MX_HOST", "direct_mx_host", "")).strip() or infer_direct_mx_host(to_email)
        mx_port_raw = resolve("DIRECT_MX_PORT", "direct_mx_port", 25)
        use_starttls = parse_bool(resolve("DIRECT_MX_USE_STARTTLS", "direct_mx_use_starttls", False), default=False)
        helo_hostname = str(
            resolve("DIRECT_MX_HELO_HOSTNAME", "direct_mx_helo_hostname", "")
        ).strip()

        try:
            mx_port = int(mx_port_raw)
        except Exception:
            mx_port = 25

        missing: List[str] = []
        if not from_email:
            missing.append("SMTP_FROM_EMAIL / from_email or --from-email")
        if not to_email:
            missing.append("SMTP_TO_EMAIL / to_email or --to-email")
        if not mx_host:
            missing.append("DIRECT_MX_HOST / direct_mx_host")

        if missing:
            return None, missing

        return (
            DirectMxSettings(
                from_email=from_email,
                to_email=to_email,
                mx_host=mx_host,
                mx_port=mx_port,
                use_starttls=use_starttls,
                helo_hostname=helo_hostname,
            ),
            [],
        )

    if delivery_mode not in {"smtp_auth", "smtp"}:
        return None, [f"Invalid delivery mode '{delivery_mode}'. Use 'smtp_auth' or 'direct_mx'."]

    host = str(resolve("SMTP_HOST", "smtp_host", "")).strip()
    port_raw = resolve("SMTP_PORT", "smtp_port", 587)
    username = str(resolve("SMTP_USERNAME", "smtp_username", "")).strip()
    password = str(resolve("SMTP_PASSWORD", "smtp_password", "")).strip()
    use_starttls = parse_bool(resolve("SMTP_USE_STARTTLS", "smtp_use_starttls", True), default=True)

    try:
        port = int(port_raw)
    except Exception:
        port = 587

    missing: List[str] = []
    if not host:
        missing.append("SMTP_HOST / smtp_host")
    if not username:
        missing.append("SMTP_USERNAME / smtp_username")
    if not password:
        missing.append("SMTP_PASSWORD / smtp_password")
    if not from_email:
        missing.append("SMTP_FROM_EMAIL / from_email or --from-email")
    if not to_email:
        missing.append("SMTP_TO_EMAIL / to_email or --to-email")

    if missing:
        return None, missing

    return (
        SmtpSettings(
            host=host,
            port=port,
            username=username,
            password=password,
            from_email=from_email,
            to_email=to_email,
            use_starttls=use_starttls,
        ),
        [],
    )


def delivery_mode_name(settings: DeliverySettings) -> str:
    if isinstance(settings, DirectMxSettings):
        return f"direct_mx({settings.mx_host}:{settings.mx_port})"
    return f"smtp_auth({settings.host}:{settings.port})"


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_checkpoints(checkpoint_path: Path) -> Tuple[int, Optional[Dict[str, Any]]]:
    if not checkpoint_path.exists():
        return 0, None
    latest: Optional[Dict[str, Any]] = None
    count = 0
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue
    return count, latest


def summarize_experiment(results_dir: Path, plots_dir: Path, experiment_pid: int) -> Tuple[str, str, List[Path], bool]:
    now = datetime.now(timezone.utc)
    run_cfg = read_json(results_dir / "run_config.json") or {}
    aggregate = read_json(results_dir / "aggregate_summary.json")

    seed_dirs = sorted([p for p in results_dir.glob("seed_*") if p.is_dir()])
    seed_ids = [p.name.split("seed_")[-1] for p in seed_dirs]

    running = is_pid_running(experiment_pid)
    status = "RUNNING" if running else "FINISHED"

    lines: List[str] = []
    lines.append(f"Generated at: {now.isoformat()}")
    lines.append(f"Experiment PID: {experiment_pid if experiment_pid > 0 else 'n/a'}")
    lines.append(f"Experiment status: {status}")
    lines.append("")

    if run_cfg:
        lines.append("Run config:")
        lines.append(json.dumps(run_cfg, indent=2, sort_keys=True))
        lines.append("")

    if seed_dirs:
        lines.append(f"Seed progress ({len(seed_dirs)} seeds):")
        for seed_dir, seed_id in zip(seed_dirs, seed_ids):
            ckpt_count, latest = read_checkpoints(seed_dir / "checkpoints.jsonl")
            summary = read_json(seed_dir / "summary.json")

            lines.append(f"- seed {seed_id}: checkpoint_rows={ckpt_count}")
            if latest:
                lines.append(
                    "  latest: "
                    f"tokens={latest.get('tokens_generated')} "
                    f"uni_delta={latest.get('unigram_max_delta')} "
                    f"bi_p95={latest.get('bigram_tv_p95')} "
                    f"tri_p95={latest.get('trigram_tv_p95')} "
                    f"converged={latest.get('converged')}"
                )
            if summary:
                lines.append(
                    "  final: "
                    f"stop_reason={summary.get('stop_reason')} "
                    f"final_tokens={summary.get('final_tokens')} "
                    f"converged={summary.get('converged')}"
                )
        lines.append("")
    else:
        lines.append("No seed directories found yet.")
        lines.append("")

    if aggregate:
        lines.append("Aggregate summary:")
        lines.append(
            json.dumps(
                {
                    k: v
                    for k, v in aggregate.items()
                    if k in {"model_id", "num_seeds", "converged_seeds", "convergence_rate", "final_tokens"}
                },
                indent=2,
                sort_keys=True,
            )
        )
        lines.append("")

    plot_files = sorted(plots_dir.rglob("*.png")) if plots_dir.exists() else []
    lines.append(f"Plots generated: {len(plot_files)}")
    for plot in plot_files:
        try:
            size_kb = plot.stat().st_size / 1024.0
        except OSError:
            size_kb = 0.0
        lines.append(f"- {plot.as_posix()} ({size_kb:.1f} KB)")

    subject = f"{status} token-stability update | seeds={len(seed_dirs)} | {now.strftime('%Y-%m-%d %H:%M UTC')}"
    body = "\n".join(lines)
    return subject, body, plot_files, running


def attach_plots(
    msg: EmailMessage,
    plot_files: List[Path],
    plots_dir: Path,
    max_attachment_mb: float,
) -> Tuple[int, Optional[Path], str]:
    if not plot_files:
        return 0, None, "No plots to attach."

    max_bytes = int(max_attachment_mb * 1024 * 1024)
    total_bytes = sum(p.stat().st_size for p in plot_files if p.exists())

    if total_bytes <= max_bytes:
        attached = 0
        for path in plot_files:
            mime_type, _ = mimetypes.guess_type(path.name)
            maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
            with path.open("rb") as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=path.name)
            attached += 1
        return attached, None, f"Attached {attached} plot(s) directly."

    # Fallback: zip all plots and attach one archive.
    tmp = tempfile.NamedTemporaryFile(prefix="token_stability_plots_", suffix=".zip", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in plot_files:
            arcname = path.relative_to(plots_dir) if path.is_relative_to(plots_dir) else path.name
            zf.write(path, arcname=str(arcname))

    zip_size = tmp_path.stat().st_size
    if zip_size <= max_bytes:
        with tmp_path.open("rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="zip", filename=tmp_path.name)
        return 1, tmp_path, (
            f"Plot images exceeded {max_attachment_mb:.1f}MB; attached zip archive "
            f"({zip_size / (1024*1024):.2f} MB) containing all plots."
        )

    # Final fallback: attach as many individual PNGs as fit.
    attached = 0
    used = 0
    for path in plot_files:
        size = path.stat().st_size
        if used + size > max_bytes:
            continue
        mime_type, _ = mimetypes.guess_type(path.name)
        maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
        with path.open("rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=path.name)
        used += size
        attached += 1

    return attached, tmp_path, (
        f"Plots are too large even as one zip ({zip_size / (1024*1024):.2f} MB). "
        f"Attached {attached}/{len(plot_files)} PNG files within {max_attachment_mb:.1f}MB."
    )


def send_email(
    settings: DeliverySettings,
    subject: str,
    body: str,
    plot_files: List[Path],
    plots_dir: Path,
    subject_prefix: str,
    max_attachment_mb: float,
    attach_enabled: bool,
) -> None:
    msg = EmailMessage()
    msg["From"] = settings.from_email
    msg["To"] = settings.to_email
    msg["Subject"] = f"{subject_prefix} {subject}"
    msg["Date"] = formatdate(localtime=False)
    msgid_domain = settings.from_email.rsplit("@", 1)[-1] if "@" in settings.from_email else None
    msg["Message-ID"] = make_msgid(domain=msgid_domain)

    # Set plain text body before adding attachments.
    msg.set_content(body)

    temp_zip: Optional[Path] = None

    if attach_enabled:
        _, temp_zip, _ = attach_plots(
            msg=msg,
            plot_files=plot_files,
            plots_dir=plots_dir,
            max_attachment_mb=max_attachment_mb,
        )

    try:
        context = ssl.create_default_context()
        if isinstance(settings, DirectMxSettings):
            smtp_kwargs: Dict[str, Any] = {"timeout": 120}
            if settings.helo_hostname:
                smtp_kwargs["local_hostname"] = settings.helo_hostname
            with smtplib.SMTP(settings.mx_host, settings.mx_port, **smtp_kwargs) as smtp:
                smtp.ehlo()
                if settings.use_starttls:
                    if not smtp.has_extn("starttls"):
                        raise RuntimeError("Remote MX does not support STARTTLS.")
                    smtp.starttls(context=context)
                    smtp.ehlo()
                smtp.send_message(msg, from_addr=settings.from_email, to_addrs=[settings.to_email])
        else:
            with smtplib.SMTP(settings.host, settings.port, timeout=120) as smtp:
                smtp.ehlo()
                if settings.use_starttls:
                    smtp.starttls(context=context)
                    smtp.ehlo()
                smtp.login(settings.username, settings.password)
                smtp.send_message(msg)
    finally:
        if temp_zip is not None and temp_zip.exists():
            temp_zip.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir).resolve()
    plots_dir = Path(args.plots_dir).resolve()
    email_cfg_path = Path(args.email_config).resolve()

    print(f"[monitor] results_dir={results_dir}")
    print(f"[monitor] plots_dir={plots_dir}")
    print(f"[monitor] email_config={email_cfg_path}")

    while True:
        try:
            settings, missing = load_delivery_settings(email_cfg_path, args.from_email, args.to_email)
            sent_this_cycle = False

            subject, body, plot_files, running = summarize_experiment(
                results_dir=results_dir,
                plots_dir=plots_dir,
                experiment_pid=args.experiment_pid,
            )

            if settings is None:
                print(
                    "[monitor] Email settings missing/invalid; skipping send this cycle. Missing fields: "
                    + ", ".join(missing)
                )
            else:
                send_email(
                    settings=settings,
                    subject=subject,
                    body=body,
                    plot_files=plot_files,
                    plots_dir=plots_dir,
                    subject_prefix=args.subject_prefix,
                    max_attachment_mb=args.max_attachment_mb,
                    attach_enabled=not args.no_attach_plots,
                )
                sent_this_cycle = True
                print(
                    f"[monitor] sent email to {settings.to_email} | plots={len(plot_files)} | "
                    f"status={'RUNNING' if running else 'FINISHED'} | mode={delivery_mode_name(settings)}"
                )

            if args.once:
                print("[monitor] --once specified; exiting.")
                return

            if args.stop_when_finished and not running and sent_this_cycle:
                print("[monitor] Experiment appears finished and --stop-when-finished is set; exiting.")
                return

            print(f"[monitor] sleeping for {args.interval_seconds} seconds")
            time.sleep(args.interval_seconds)

        except KeyboardInterrupt:
            print("[monitor] interrupted; exiting.")
            return
        except Exception as exc:
            print(f"[monitor] cycle failed: {exc}")
            if args.once:
                raise
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
