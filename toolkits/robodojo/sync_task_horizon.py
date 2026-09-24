#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Print RoboDojo task horizons and OpenDM support from the local checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_PATH))

from rlinf.envs.robodojo.task_inventory import (  # noqa: E402
    get_task_horizon,
    list_task_records,
    opendm_supported_task_names,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-path", default=None)
    parser.add_argument("--task", default=None, help="Print step_lim for one task")
    parser.add_argument("--opendm-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.task:
        horizon = get_task_horizon(args.task, args.robodojo_path)
        print(horizon)
        return 0

    records = list_task_records(args.robodojo_path)
    if args.opendm_only:
        names = opendm_supported_task_names(args.robodojo_path)
        if args.json:
            print(json.dumps(names, indent=2))
        else:
            for name in names:
                print(name)
        return 0

    if args.json:
        print(json.dumps(records, indent=2))
        return 0

    print(f"{'task':40} {'step_lim':>8} {'opendm':>8} {'robot':>28}")
    for record in records:
        print(
            f"{record['name']:40} {str(record['step_lim'] or '-'):>8} "
            f"{str(record['opendm_supported']):>8} {record['robot_config']:>28}"
        )
    supported = opendm_supported_task_names(args.robodojo_path)
    print(f"\nOpenDM-supported dual_x5 tasks: {len(supported)}")
    skipped = [r["name"] for r in records if r["competition"]]
    if skipped:
        print("Skipped competition / non-dual_x5:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
