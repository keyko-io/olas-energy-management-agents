#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------


"""Updates fetched agent with correct config"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def main() -> None:
    """Main"""
    load_dotenv()

    with open(Path("peaq_agent", "aea-config.yaml"), "r", encoding="utf-8") as file:
        config = list(yaml.safe_load_all(file))
        if os.getenv("COMBINDER_API_URL"):
            config[1]["models"]["params"]["args"]["combinder_api_url"] = f"${{str:{os.getenv('COMBINDER_API_URL')}}}"

        if os.getenv("MODEL_API_URL"):
            config[1]["models"]["params"]["args"]["model_api_url"] = f"${{str:{os.getenv('MODEL_API_URL')}}}"

        if os.getenv("COMBINDER_API_KEY"):
            config[1]["models"]["params"]["args"]["combinder_api_key"] = f"${{str:{os.getenv('COMBINDER_API_KEY')}}}"

        if os.getenv("MODEL_API_KEY"):
            config[1]["models"]["params"]["args"]["model_api_key"] = f"${{str:{os.getenv('MODEL_API_KEY')}}}"

        if os.getenv("PREFILL_DATA"):
            config[1]["models"]["params"]["args"]["prefill_data"] = f"${{bool:{os.getenv('PREFILL_DATA')}}}"

        # Ledger RPCs
        if os.getenv("ETHEREUM_LEDGER_RPC"):
            config[2]["config"]["ledger_apis"]["ethereum"][
                "address"
            ] = f"${{str:{os.getenv('ETHEREUM_LEDGER_RPC')}}}"

        if os.getenv("GNOSIS_LEDGER_RPC"):
            config[2]["config"]["ledger_apis"]["gnosis"][
                "address"
            ] = f"${{str:{os.getenv('GNOSIS_LEDGER_RPC')}}}"


    with open(Path("peaq_agent", "aea-config.yaml"), "w", encoding="utf-8") as file:
        yaml.dump_all(config, file, sort_keys=False)


if __name__ == "__main__":
    main()
