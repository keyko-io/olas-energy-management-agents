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

"""This package contains the rounds of SendAPIDataAbciApp."""

from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectSameUntilAllRound,
    CollectionRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    get_name
)
from packages.keyko.skills.send_api_data_abci.payloads import (
    ProjectDataSubmissionDecisionPayload,
    ProjectDataSubmissionPayload,
    AgentDataSubmissionPayload,
)


class SendApiDataEvent(Enum):
    """SendAPIDataAbciApp Events"""

    DONE = "done"
    ROUND_TIMEOUT = "round_timeout"
    SEND_PROJECT_DATA = "send_project_data"
    NO_MAJORITY = "no_majority"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    def _get_deserialized(self, key: str) -> DeserializedCollection:
        """Strictly get a collection and return it deserialized."""
        serialized = self.db.get_strict(key)
        return CollectionRound.deserialize_collection(serialized)
    
    @property
    def is_project_data_submitted(self) -> Optional[bool]:
        """Check if the project data has already been submitted."""
        return self.db.get("is_project_data_submitted", None)
    
    @property
    def history_slug_timestamp(self) -> Optional[str]:
        """Unix timestamp for the first agent data submission to the DB."""
        return self.db.get("history_slug_timestamp", None)


class ProjectDataSubmissionDecisionRound(CollectSameUntilThresholdRound):
    """ProjectDataSubmissionDecisionRound"""
    
    payload_class = ProjectDataSubmissionDecisionPayload
    synchronized_data_class = SynchronizedData
    selection_key = (
        get_name(SynchronizedData.is_project_data_submitted),
    )
    ERROR_PAYLOAD = "{}"

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, SendApiDataEvent]]:
        """Process the end of the block."""

        if self.threshold_reached:
            synchronized_data = self.synchronized_data.update(
                is_project_data_submitted=self.most_voted_payload_values[1],
                history_slug_timestamp=self.most_voted_payload_values[2],
                synchronized_data_class=SynchronizedData,
            )
            event = SendApiDataEvent(self.most_voted_payload)
            return synchronized_data, event

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, SendApiDataEvent.NO_MAJORITY

        return None


class ProjectDataSubmissionRound(CollectSameUntilAllRound):
    """ProjectDataSubmissionRound"""

    payload_class = ProjectDataSubmissionPayload
    synchronized_data_class = SynchronizedData
    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, SendApiDataEvent]]:
        """Process the end of the block."""
        if self.collection_threshold_reached:
            synchronized_data = self.synchronized_data.update(
                participants=tuple(sorted(self.collection)),
                synchronized_data_class=SynchronizedData,
            )
            return synchronized_data, SendApiDataEvent.DONE
        return None


class AgentDataSubmissionRound(CollectSameUntilAllRound):
    """AgentDataSubmissionRound"""

    payload_class = AgentDataSubmissionPayload
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, SendApiDataEvent]]:
        """Process the end of the block."""
        if self.collection_threshold_reached:
            synchronized_data = self.synchronized_data.update(
                participants=tuple(sorted(self.collection)),
                synchronized_data_class=SynchronizedData,
            )
            return synchronized_data, SendApiDataEvent.DONE
        return None


class FinishedAgentDataSubmissionRound(DegenerateRound):
    """FinishedAgentDataSubmissionRound"""


class SendAPIDataAbciApp(AbciApp[SendApiDataEvent]):
    """SendAPIDataAbciApp"""

    initial_round_cls: AppState = ProjectDataSubmissionDecisionRound
    initial_states: Set[AppState] = {
        ProjectDataSubmissionDecisionRound,
    }
    transition_function: AbciAppTransitionFunction = {
        ProjectDataSubmissionDecisionRound: {
            SendApiDataEvent.DONE: AgentDataSubmissionRound,
            SendApiDataEvent.SEND_PROJECT_DATA: ProjectDataSubmissionRound,
            SendApiDataEvent.ROUND_TIMEOUT: ProjectDataSubmissionDecisionRound,
        },
        ProjectDataSubmissionRound: {
            SendApiDataEvent.DONE: AgentDataSubmissionRound,
            SendApiDataEvent.ROUND_TIMEOUT: ProjectDataSubmissionRound,
        },
        AgentDataSubmissionRound: {
            SendApiDataEvent.DONE: FinishedAgentDataSubmissionRound,
            SendApiDataEvent.ROUND_TIMEOUT: AgentDataSubmissionRound,
        },
        FinishedAgentDataSubmissionRound: {}
    }
    final_states: Set[AppState] = {
        FinishedAgentDataSubmissionRound,
    }
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset(
        [get_name(SynchronizedData.is_project_data_submitted), get_name(SynchronizedData.history_slug_timestamp)]
    )
    db_pre_conditions: Dict[AppState, Set[str]] = {
        ProjectDataSubmissionDecisionRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedAgentDataSubmissionRound: set(),
    }