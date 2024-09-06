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

"""This package contains round behaviours of SendAPIDataAbciApp."""

from abc import ABC
from datetime import datetime, timedelta
from typing import Generator, List, Set, Type, cast, Optional
import re
import json

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.keyko.skills.send_api_data_abci.models import ProjectDBData, SendAPIDataParams, SharedState
from packages.keyko.skills.send_api_data_abci.payloads import (
    ProjectDataSubmissionDecisionPayload, 
    ProjectDataSubmissionPayload, 
    AgentDataSubmissionPayload, 
)

from packages.keyko.skills.send_api_data_abci.rounds import (
    ProjectDataSubmissionDecisionRound,
    ProjectDataSubmissionRound,
    AgentDataSubmissionRound,
    SendAPIDataAbciApp,
    SendApiDataEvent,
    SynchronizedData
)

WaitableConditionType = Generator[None, None, bool]

class SendAPIDataBaseBehaviour(BaseBehaviour, ABC):  # pylint: disable=too-many-ancestors
    """Base behaviour for the send_api_data_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> SendAPIDataParams:
        """Return the params."""
        return cast(SendAPIDataParams, super().params)

    @property
    def local_state(self) -> SharedState:
        """Return the state."""
        return cast(SharedState, self.context.state)


class ProjectDataSubmissionDecisionBehaviour(SendAPIDataBaseBehaviour):  # pylint: disable=too-many-ancestors
    """ProjectDataSubmissionDecisionBehaviour"""

    matching_round: Type[AbstractRound] = ProjectDataSubmissionDecisionRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            history_slug_timestamp = self.synchronized_data.history_slug_timestamp
            if history_slug_timestamp is None:
                history_slug_timestamp = str(int(datetime.now().timestamp()))
            
            
            if self.synchronized_data.is_project_data_submitted is not True:
                self.context.logger.info(
                    f"ProjectDataSubmissionDecisionBehaviour: Project data is not submitted."
                )
                event = SendApiDataEvent.SEND_PROJECT_DATA.value
            else:
                self.context.logger.info(
                    f"ProjectDataSubmissionDecisionBehaviour: Project data has already been submitted."
                )
                event=SendApiDataEvent.DONE.value
                
            sender = self.context.agent_address
            payload = ProjectDataSubmissionDecisionPayload(
                sender=sender,
                event=event, 
                is_project_data_submitted=True,
                history_slug_timestamp=history_slug_timestamp
            )
            self.context.logger.info(
                f"ProjectDataSubmissionDecisionBehaviour: Agent with address: {self.context.agent_address} has decided to submit the project data: {event}."
            )
            self.context.logger.info(f"payload: {payload}")
            self.context.logger.info(f"Sync data: {self.synchronized_data}")
            

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

        
class ProjectDataSubmissionBehaviour(SendAPIDataBaseBehaviour):  # pylint: disable=too-many-ancestors
    """ProjectDataSubmissionBehaviour"""

    matching_round: Type[AbstractRound] = ProjectDataSubmissionRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            participants = sorted(self.synchronized_data.participants)
            self.context.logger.info(
                f"ProjectDataSubmissionBehaviour: Agent which will submit project data={participants[0]}."
            )
            
            if (self.context.agent_address == participants[0]):
                project_db_data: ProjectDBData = self.params.project_db_data.to_json()
                olas_api_x_api_key = self.params.olas_api_x_api_key
                
                content=json.dumps({
                    "project": project_db_data["project"],
                    "fsmSuite": project_db_data["fsm_suite"],
                    "instance": project_db_data["instance"],
                    "networkId": 0
                }).encode("utf-8")
                yield from self.get_http_response(
                    method="POST",
                    url=self.params.olas_api_url_project_data,
                    headers={
                        "x-api-key": olas_api_x_api_key,
                        "Content-Type": "application/json",
                    },
                    content=content,
                )
                
            sender = self.context.agent_address
            payload = ProjectDataSubmissionPayload(
                sender=sender, 
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class AgentDataSubmissionBehaviour(SendAPIDataBaseBehaviour):  # pylint: disable=too-many-ancestors
    """AgentDataSubmissionBehaviour"""

    matching_round: Type[AbstractRound] = AgentDataSubmissionRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            self.context.logger.info(
                f"AgentDataSubmissionBehaviour: Agent with address: {self.context.agent_address} has submitted the current cycle data to the API."
            )
            
            history_slug_timestamp = self.synchronized_data.history_slug_timestamp
            olas_api_x_api_key = self.params.olas_api_x_api_key
            project_db_data: ProjectDBData = self.params.project_db_data.to_json()
            
            name = self.params.agent_db_name
            description = self.params.agent_db_description
            slug = self.params.agent_db_slug
            
            content=json.dumps({
                "name": name,
                "description": description,
                "slug": slug,
                "instanceSlug": project_db_data["instance"]["slug"],
                "historySlug": project_db_data["instance"]["slug"] + "-" + slug + '-' + history_slug_timestamp,
                "walletAddress": self.context.agent_address,
            }).encode("utf-8")
            
            response = yield from self.get_http_response(
                method="POST",
                url=self.params.olas_api_url_agent_data,
                headers={
                    "x-api-key": olas_api_x_api_key,
                    "Content-Type": "application/json",
                },
                content=content,
            )
            
            self.context.logger.info(
                f"==============================================="
            )
            self.context.logger.info(
                f"{response.status_code}"
            )
            self.context.logger.info(
                f"{response.body}"
            )
            self.context.logger.info(
                f"==============================================="
            )
            
            sender = self.context.agent_address
            payload = AgentDataSubmissionPayload(
                sender=sender
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class SendAPIDataRoundBehaviour(AbstractRoundBehaviour):
    """SendAPIDataRoundBehaviour"""

    initial_behaviour_cls = ProjectDataSubmissionDecisionBehaviour
    abci_app_cls = SendAPIDataAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [  # type: ignore
        ProjectDataSubmissionDecisionBehaviour,
        ProjectDataSubmissionBehaviour,
        AgentDataSubmissionBehaviour,
    ]
