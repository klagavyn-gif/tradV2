import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request


UTC = datetime.timezone.utc


def parse_iso(value):
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def parse_state_time(value):
    try:
        return datetime.datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except Exception:
        return None


def load_state(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    thread_id = os.environ.get("TELEGRAM_THREAD_ID")
    if not token or not chat_id:
        print("[heartbeat] telegram secrets missing; skip send")
        return False
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    if thread_id:
        payload["message_thread_id"] = thread_id
    data = json.dumps(payload).encode("utf-8")
    url = "https://api.telegram.org/bot{}/sendMessage".format(token)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                if 200 <= int(getattr(response, "status", 0)) < 300:
                    return True
        except Exception as exc:
            print("[heartbeat] telegram send attempt {} failed: {}".format(attempt + 1, exc))
            time.sleep(2)
    return False


def recent_runs(workflow):
    result = subprocess.run(
        [
            "gh", "run", "list",
            "--workflow", workflow,
            "--event", "workflow_dispatch",
            "--limit", "5",
            "--json", "databaseId,createdAt,conclusion,headSha,url",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("[heartbeat] gh run list failed: {}".format(result.stderr.strip()))
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        return None


def evaluate_freshness(runs, *, now, max_age_minutes):
    if not runs:
        return True, "ไม่สามารถอ่านประวัติ workflow run ล่าสุดได้"
    epoch = datetime.datetime.min.replace(tzinfo=UTC)
    latest = max(runs, key=lambda row: parse_iso(row.get("createdAt")) or epoch)
    created = parse_iso(latest.get("createdAt"))
    conclusion = str(latest.get("conclusion") or "").strip()
    run_id = latest.get("databaseId")
    age_minutes = (now - created).total_seconds() / 60.0 if created else None
    if age_minutes is None:
        return True, "อ่านเวลาของ run ล่าสุดไม่ได้"
    if age_minutes > max_age_minutes:
        return True, "run ล่าสุดเก่าเกินกำหนด id={} age={:.1f} นาที conclusion={}".format(
            run_id, age_minutes, conclusion or "running"
        )
    if conclusion == "success":
        return False, "run ล่าสุดปกติ id={} age={:.1f} นาที".format(run_id, age_minutes)
    if conclusion == "":
        return False, "run ล่าสุดกำลังทำงาน id={} age={:.1f} นาที".format(run_id, age_minutes)
    return True, "run ล่าสุดล้มเหลว id={} conclusion={} age={:.1f} นาที".format(
        run_id, conclusion, age_minutes
    )


def main():
    max_age_minutes = float(os.environ.get("HEARTBEAT_MAX_AGE_MINUTES") or 25)
    repeat_hours = float(os.environ.get("HEARTBEAT_REPEAT_HOURS") or 6)
    workflow = os.environ.get("HEARTBEAT_WORKFLOW") or "main.yml"
    state_path = pathlib.Path(
        os.environ.get("HEARTBEAT_STATE_PATH") or ".data/heartbeat/heartbeat_state.json"
    )
    force_alert = str(os.environ.get("HEARTBEAT_FORCE_ALERT") or "").strip().lower() == "true"

    now = datetime.datetime.now(UTC)
    now_text = now.strftime("%Y-%m-%d %H:%M:%S")
    runs = recent_runs(workflow)
    problem, detail = evaluate_freshness(runs, now=now, max_age_minutes=max_age_minutes)
    if problem:
        # Guard against a transient stale run list: the GitHub API occasionally
        # returns an old run as "latest" even while fresh runs exist, which
        # produces a false "pipeline stale" alert. Re-query once and trust a
        # healthy second result before alerting.
        time.sleep(2)
        runs2 = recent_runs(workflow)
        problem2, detail2 = evaluate_freshness(runs2, now=now, max_age_minutes=max_age_minutes)
        if not problem2:
            problem = False
            detail = "{} (recheck ปกติ กัน false positive)".format(detail2)
    if force_alert:
        problem = True
        detail = "forced heartbeat alert for testing ({})".format(detail)

    state = load_state(state_path)
    was_alerting = bool(state.get("alerting"))
    last_alert_at = parse_state_time(state.get("last_alert_at"))
    sent = False

    if problem:
        should_send = (
            not was_alerting
            or last_alert_at is None
            or (now - last_alert_at).total_seconds() > repeat_hours * 3600
        )
        if should_send:
            message = (
                "<b>tradV2 heartbeat: ระบบแจ้งเตือนผิดปกติ</b>\n"
                "{}\nเวลา: {} UTC".format(detail, now_text)
            )
            sent = send_telegram(message)
            if sent:
                state["alerting"] = True
                state["last_alert_at"] = now_text
        else:
            print("[heartbeat] problem persists; within repeat window")
    else:
        if was_alerting:
            message = (
                "<b>tradV2 heartbeat: ระบบกลับมาปกติ</b>\n"
                "{}\nเวลา: {} UTC".format(detail, now_text)
            )
            send_telegram(message)
        state["alerting"] = False

    state["last_status"] = "problem" if problem else "ok"
    state["last_detail"] = detail
    state["last_checked_at"] = now_text
    save_state(state_path, state)
    print("[heartbeat] problem={} sent={} detail={}".format(problem, sent, detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
