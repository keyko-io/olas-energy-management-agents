# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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
"""Contains the job definitions"""

from typing import Any, Dict, Optional, Tuple
import requests

PREDICT_ENDPOINT = "https://peaq-service.hmbam5qcfhh70.eu-central-1.cs.amazonlightsail.com/predict"

def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    print('Running the task')
    api_key = kwargs["api_keys"]["peaq_service"]
    data = kwargs["data"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(PREDICT_ENDPOINT, json=data, headers=headers)
        print(response.status_code)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}", None, None, None
    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None

    return response_json, data, None, None

if __name__ == "__main__":
    run(
        api_keys={"peaq_service": "your_api_key"},
        data={"key": "value"}
    )