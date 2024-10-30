# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
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

"""This package contains round behaviours of PeaqAbciApp."""

from abc import ABC
from typing import Generator, Set, Type, cast
import json
from datetime import datetime
import urllib

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)

from packages.keyko.skills.peaq_abci.models import Params
from packages.keyko.skills.peaq_abci.rounds import (
    SynchronizedData,
    PeaqAbciApp,
    CollectDataRound,
    DeviceInteractionRound,
    QueryModelRound,
)
from packages.keyko.skills.peaq_abci.rounds import (
    CollectDataPayload,
    DeviceInteractionPayload,
    QueryModelPayload,
)


LOG_FILE_PATH = "peaq_agent.log"

def log_message(message: str, data: str = ""):
    """Write a log message to the log file."""
    with open(LOG_FILE_PATH, "a") as log_file:
        log_entry = f"{datetime.now().isoformat()} - {message}"
        if data:
            log_entry += f" - Data: {data}"
        log_file.write(log_entry + "\n")

class PeaqBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the peaq_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)

class CollectDataBehaviour(PeaqBaseBehaviour):
    """CollectDataBehaviour"""

    matching_round: Type[AbstractRound] = CollectDataRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        log_message("Collecting data from Combinder API")

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            solar_data = yield from self.get_combinder_past_data(self.params.solar_device_id)
            consumption_data = yield from self.get_combinder_past_data(self.params.ac_device_id)

            combined_data = [
                {
                    'cet_cest_timestamp': self.convert_timestamp(solar[0]), 
                    'pv': solar[1] if solar[1] is not None else 0.0, 
                    'grid_import': consumption[1] if consumption[1] is not None else 0.0
                }
                for solar, consumption in zip(solar_data, consumption_data)
            ]


            sender = self.context.agent_address
            log_message("Combinder API data collected")
            self.context.logger.info(f"CollectDataBehaviour: Combined data: {combined_data}")
            payload = CollectDataPayload(sender=sender, prosumer_data=combined_data)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_combinder_past_data(self, device_id) -> Generator[None, None, str]:
        """
        Get the data from Combinder API which contains the power production and consumption data.
        """
        log_message(f"Retrieving last data from Combinder API for device {device_id}")
        url = f"{self.params.combinder_api_url}/devices/{device_id}/last_60min"
        response = yield from self.get_http_response(
            method="GET",
            url=url,
            headers={"Authorization": f"Bearer {self.params.combinder_api_key}"}
        )
        if response.status_code != 200:
            self.context.logger.error(
                f"APICheckBehaviour: url: {url}"
                f"APICheckBehaviour: Could not retrieve data from Combinder API. "
                f"APICheckBehaviour: Received status code {response.status_code}."
            )
            self.context.logger.error(f"APICheckBehaviour: Response body: {response.body}")
            self.context.logger.error(f"APICheckBehaviour: Response headers: {response.headers}")
            return ""
        
        data = json.loads(response.body)['result']
        log_message(f"Data retrieved from Combinder API for device {device_id}")
        
        return data
    
    def convert_timestamp(self, timestamp):
        # Parsear el timestamp incluyendo la zona horaria
        dt = datetime.fromisoformat(timestamp)
        # Retornar el timestamp en el formato deseado
        return dt.strftime('%Y-%m-%d %H:%M:%S')

class QueryModelBehaviour(PeaqBaseBehaviour):
    """QueryModelBehaviour"""

    matching_round: Type[AbstractRound] = QueryModelRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        # Step 2: Query the model to get the prediction
        log_message("Gathering results from the model")

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            model_prediction = yield from self.query_model()
            action_str = "on" if model_prediction else "off"
            log_message(f"Model prediction obtained: Turn {action_str} device")
            self.context.logger.info(f"QueryModelBehaviour: Model prediction: {model_prediction}")

            sender = self.context.agent_address
            payload = QueryModelPayload(sender=sender, prediction_class=model_prediction)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def query_model(self):
        # Query the model with synchronized data
        url = self.params.model_api_url + f"/predict"

        http_params = {
            "model_id": self.params.model_id,
            "user_input": {
                "data": self.synchronized_data.prosumer_data
            },
            "system_prompt": "none",
            "temperature": self.params.temperature,
            "max_tokens" : self.params.max_tokens
        }
        response = yield from self.get_http_response(
            method="POST",
            url=url,
            headers={
                "Authorization": f"Bearer {self.params.model_api_key}",
                "Content-Type": "application/json"
            },
            content=json.dumps(http_params).encode('utf-8')
        )

        if response.status_code != 200:
            self.context.logger.error(
                f"QueryModelBehaviour: url: {url}"
                f"QueryModelBehaviour: Could not predict data from Combinder API. "
                f"QueryModelBehaviour: Received status code {response.status_code}."
            )
            self.context.logger.error(f"QueryModelBehaviour: Response body: {response.body}")
            self.context.logger.error(f"QueryModelBehaviour: Response headers: {response.headers}")
            return -1

        data = json.loads(response.body)

        return data["prediction_class"]

class DeviceInteractionBehaviour(PeaqBaseBehaviour):
    """DeviceInteractionBehaviour"""

    matching_round: Type[AbstractRound] = DeviceInteractionRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            device_action = bool(self.synchronized_data.last_prediction_class)
            action_str = "on" if device_action else "off"
            log_message(f"Switching {action_str} Air Conditioning Unit with device id {self.params.ac_device_id}")

            ret = yield from self.control_device(self.params.ac_device_id, device_action)
            sender = self.context.agent_address
            # If ret["result"] is {} then the device was controlled successfully
            success = True if ret["result"] == {} else False
            message = ret["message"] if "success" in ret and ret["success"] == False else "Device controlled successfully."
            log_message("Device control response", data=message)
            log_message("Put agent into sleep for 1 minute...")
        
            payload = DeviceInteractionPayload(sender=sender, success=success, message=message)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()
        
    def control_device(self, device_id, device_action):
        # Control the device based on the prediction
        url = f"{self.params.combinder_api_url}/devices/{device_id}/switch"
        response = yield from self.get_http_response(
            method="POST",
            url=url,
            headers={
                "Authorization": f"Bearer {self.params.combinder_api_key}",
                "Content-Type": "application/x-www-form-urlencoded"
            },
            content=urllib.parse.urlencode({
                "on": "true" if device_action else "false"
            }).encode('utf-8')
        )
        if response.status_code != 200:
            self.context.logger.error(
                f"DeviceInteractionBehaviour: url: {url}\n"
                f"DeviceInteractionBehaviour: Could not control device from Combinder API.\n"
                f"DeviceInteractionBehaviour: Received status code {response.status_code}.\n"
            )
            self.context.logger.error(f"DeviceInteractionBehaviour: Response body: {response.body}")
            self.context.logger.error(f"DeviceInteractionBehaviour: Response headers: {response.headers}")
            return {
                "success": False,
                "message": f"Could not control device. (Status code: {response.status_code})"
            }
        data = json.loads(response.body)

        return data

class PeaqRoundBehaviour(AbstractRoundBehaviour):
    """PeaqRoundBehaviour"""

    initial_behaviour_cls = CollectDataBehaviour
    abci_app_cls = PeaqAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [
        CollectDataBehaviour,
        DeviceInteractionBehaviour,
        QueryModelBehaviour,
    ]
