# -*- coding: utf-8 -*-
"""
wipe_data.py — Dev CLI tool 清空 DB 的 ML state (給開發測試用,不對外暴露)

用法 (在專案根目錄執行):
    python tools/wipe_data.py --all              # 清掉所有 user 的 ML 資料
    python tools/wipe_data.py --user 2           # 只清 user_id=2 的資料
    python tools/wipe_data.py --user me@x.com    # 用 email 找 user
    python tools/wipe_data.py --all --include-users  # 連 user 表也清 (重置帳號)

預覽不執行:
    python tools/wipe_data.py --all --dry-run

不會動的:
    - users 表的結構 (除非加 --include-users)
    - .env / 程式碼 / 設定
"""
from __future__ import annotations

import argparse
import os
import sys

# 把專案根目錄加進 sys.path,才能 import api.*
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 載入 .env (本地開發用)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from api.auth.db import (
    Dataset, Preprocessor, Model, TrainingRun, PredictionArtifact, User,
    SessionLocal, engine,
)


TABLES_CHILD_TO_PARENT = [
    ("prediction_artifacts", PredictionArtifact),
    ("models",               Model),
    ("training_runs",        TrainingRun),
    ("preprocessors",        Preprocessor),
    ("datasets",             Dataset),
]


def _resolve_user(db, user_arg: str) -> User | None:
    """user_arg 可以是整數 id 或 email。"""
    try:
        uid = int(user_arg)
        return db.query(User).filter_by(id=uid).first()
    except ValueError:
        return db.query(User).filter_by(email=user_arg).first()


def main():
    p = argparse.ArgumentParser(
        description="清空 ML 資料 (dev tool,給開發者排除舊資料干擾用)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="清掉所有 user 的 ML 資料")
    g.add_argument("--user", help="只清指定 user (id 或 email)")
    p.add_argument("--include-users", action="store_true",
                   help="連 users 表也清掉 (注意:會把帳號刪除,登入要重新註冊)")
    p.add_argument("--dry-run", action="store_true", help="只預覽筆數,不真的刪")
    args = p.parse_args()

    db = SessionLocal()
    try:
        # === 預覽 ===
        print(f"\n{'='*60}")
        print(f"連線目標: {engine.url}")
        print(f"{'='*60}\n")

        if args.all:
            target_label = "所有 user"
            user_filter = None
        else:
            user = _resolve_user(db, args.user)
            if user is None:
                print(f"❌ 找不到 user: {args.user}")
                sys.exit(1)
            target_label = f"user_id={user.id} ({user.email})"
            user_filter = user.id

        print(f"目標範圍: {target_label}")
        print(f"動作: {'預覽 (不刪)' if args.dry_run else '實際刪除'}\n")

        print(f"{'Table':<25} {'筆數':>10}")
        print("-" * 36)
        plans = []
        for name, model in TABLES_CHILD_TO_PARENT:
            q = db.query(model)
            if user_filter is not None:
                q = q.filter_by(user_id=user_filter)
            n = q.count()
            plans.append((name, model, q, n))
            print(f"{name:<25} {n:>10}")

        if args.include_users:
            q_users = db.query(User)
            if user_filter is not None:
                q_users = q_users.filter_by(id=user_filter)
            n_users = q_users.count()
            print(f"{'users':<25} {n_users:>10}  ⚠️ (會被刪)")
            plans.append(("users", User, q_users, n_users))

        total = sum(n for _, _, _, n in plans)
        print("-" * 36)
        print(f"{'合計':<25} {total:>10} 筆")

        if args.dry_run:
            print("\n--dry-run 模式,不執行刪除。")
            return

        if total == 0:
            print("\n沒有資料要刪。")
            return

        # === 確認 ===
        print()
        confirm = input(f"⚠️  確定要刪掉以上 {total} 筆資料?(輸入 'YES' 大寫): ")
        if confirm != "YES":
            print("取消。")
            return

        # === 實際刪除 (孩子 → 父) ===
        print()
        deleted = {}
        for name, model, q, _ in plans:
            d = q.delete(synchronize_session=False)
            deleted[name] = d
            print(f"  ✓ 刪除 {name}: {d} 筆")
        db.commit()
        print(f"\n✅ 完成。總計刪除 {sum(deleted.values())} 筆。")

        _print_local_clear_hint()
    finally:
        db.close()


def _print_local_clear_hint():
    """
    DB 清空後,瀏覽器 localStorage 仍可能殘留舊訓練歷史 / 表單設定。
    這支 CLI 在伺服器端執行,碰不到瀏覽器,所以這裡做兩件事:
      1. 說明:舊 UI 的 hydrate 已改為「DB 為唯一真相」,
         登入狀態下重新整理頁面 (F5) 就會自動清掉本機殘留歷史。
      2. 對於「已經開著、不會重新整理」的分頁,提供一段 console 指令手動清。
    """
    snippet = (
        "Object.keys(localStorage)"
        ".filter(k => k.startsWith('automl_training_history') "
        "|| k.startsWith('automl_latest_full') "
        "|| k.startsWith('automl_exp_form'))"
        ".forEach(k => localStorage.removeItem(k)); location.reload();"
    )
    print("\n" + "=" * 60)
    print("🧹 清 local (瀏覽器端)")
    print("=" * 60)
    print("DB 已清空。本機 (localStorage) 的清除方式:")
    print("  • 已登入 → 重新整理頁面 (F5) 即自動清乾淨 (DB 為唯一真相)。")
    print("  • 未登入 / 想立即清「已開著的分頁」→ 在瀏覽器 Console 貼上:\n")
    print("    " + snippet)
    print()


if __name__ == "__main__":
    main()
