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

"""This package contains the rounds of PeaqAbciApp."""

from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, cast
import json

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AbstractRound,
    AppState,
    BaseSynchronizedData,
    DegenerateRound,
    EventToTimeout,
    get_name,
    CollectSameUntilThresholdRound
)

from packages.keyko.skills.peaq_abci.payloads import (
    CollectDataPayload,
    DeviceInteractionPayload,
    QueryModelPayload,
    RegistrationPayload,
    ResetAndPausePayload,
)

class Event(Enum):
    """PeaqAbciApp Events"""

    TRANSACT = "transact"
    ROUND_TIMEOUT = "round_timeout"
    NO_TRANSACT = "no_transact"
    DONE = "done"
    ERROR = "error"
    RESET_TIMEOUT = "reset_timeout"
    NOT_ENOUGH_DATA = "not_enough_data"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """
    @property
    def prosumer_data(self) -> List:
        """Get the prosumer data list."""
        return self.db.get("prosumer_data", [])
    
    @property
    def last_prediction_class(self) -> int:
        """Get the last prediction class."""
        return self.db.get("last_prediction_class", -1)

class RegistrationRound(CollectSameUntilThresholdRound):
    """RegistrationRound"""

    payload_class = RegistrationPayload
    synchronized_data_class = SynchronizedData
    payload_sent = False

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""

        if not self.payload_sent:
            return None
        
        if len(self.synchronized_data.prosumer_data) >= 59:
            return self.synchronized_data, Event.DONE
        return None

    def check_payload(self, payload: CollectDataPayload) -> None:
        """Check payload."""
        return

    def process_payload(self, payload: CollectDataPayload) -> None:
        """Process payload."""
        self.synchronized_data.update(
            participants=tuple(sorted(self.collection)),
            prosumer_data=payload.prosumer_data[-59:],
            synchronized_data_class=SynchronizedData,
        )
        self.payload_sent = True
        return

class CollectDataRound(CollectSameUntilThresholdRound):
    """CollectDataRound"""

    payload_class = CollectDataPayload
    synchronized_data_class = SynchronizedData
    payload_sent = False

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if not self.payload_sent:
            return None
        
        self.payload_sent = False
        
        if len(self.synchronized_data.prosumer_data) >= 60:
            return self.synchronized_data, Event.DONE
    
        return self.synchronized_data, Event.NOT_ENOUGH_DATA

    def check_payload(self, payload: CollectDataPayload) -> None:
        """Check payload."""
        return

    def process_payload(self, payload: CollectDataPayload) -> None:
        """Process payload."""
        prosumer_data = self.synchronized_data.prosumer_data
        prosumer_data.append(payload.prosumer_data)
        self.synchronized_data.update(
            prosumer_data=prosumer_data[-60:],
            synchronized_data_class=SynchronizedData,
        )
        self.payload_sent = True
        return


class QueryModelRound(AbstractRound):
    """QueryModelRound"""

    payload_class = QueryModelPayload
    synchronized_data_class = SynchronizedData
    err = False
    status_change = False
    payload_sent = False

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if not self.payload_sent:
            return None
        self.payload_sent = False
        
        if self.err:
            return self.synchronized_data, Event.ERROR
        if self.status_change:
            return self.synchronized_data, Event.TRANSACT
        return self.synchronized_data, Event.NO_TRANSACT
        

    def check_payload(self, payload: QueryModelPayload) -> None:
        """Check payload."""
        return

    def process_payload(self, payload: QueryModelPayload) -> None:
        """Process payload."""
        self.payload_sent = True
        self.err = False
        self.status_change = False
        if payload.prediction_class == -1:
            self.err = True
            return
        
        last_prediction_class = self.synchronized_data.last_prediction_class
        if payload.prediction_class == last_prediction_class:
            return
        
        self.status_change = True
        self.synchronized_data.update(
            last_prediction_class=payload.prediction_class,
            synchronized_data_class=SynchronizedData,
        )
        return


class DeviceInteractionRound(AbstractRound):
    """DeviceInteractionRound"""

    payload_class = DeviceInteractionPayload
    synchronized_data_class = SynchronizedData
    payload_sent = False
    interaction_success = False

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if not self.payload_sent:
            return None
        
        if self.interaction_success:
            return self.synchronized_data, Event.DONE
        
        return self.synchronized_data, Event.ERROR


    def check_payload(self, payload: DeviceInteractionPayload) -> None:
        """Check payload."""
        return

    def process_payload(self, payload: DeviceInteractionPayload) -> None:
        """Process payload."""
        self.payload_sent = True
        self.interaction_success = payload.success

class ResetAndPauseRound(AbstractRound):
    """ResetAndPauseRound"""

    payload_class = ResetAndPausePayload
    synchronized_data_class = SynchronizedData
    payload_sent = False

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if not self.payload_sent:
            return None
        self.payload_sent = False
        return self.synchronized_data, Event.DONE


    def check_payload(self, payload: ResetAndPausePayload) -> None:
        """Check payload."""
        return

    def process_payload(self, payload: ResetAndPausePayload) -> None:
        """Process payload."""
        self.payload_sent = True

class PeaqAbciApp(AbciApp[Event]):
    """PeaqAbciApp"""

    initial_round_cls: AppState = RegistrationRound
    initial_states: Set[AppState] = {RegistrationRound}
    transition_function: AbciAppTransitionFunction = {
        CollectDataRound: {
            Event.DONE: QueryModelRound,
            Event.NOT_ENOUGH_DATA: ResetAndPauseRound,
            Event.ROUND_TIMEOUT: CollectDataRound
        },
        QueryModelRound: {
            Event.NO_TRANSACT: ResetAndPauseRound,
            Event.TRANSACT: DeviceInteractionRound,
            Event.ERROR: ResetAndPauseRound
        },
        ResetAndPauseRound: {
            Event.DONE: CollectDataRound,
            Event.RESET_TIMEOUT: RegistrationRound
        },
        RegistrationRound: {
            Event.DONE: CollectDataRound
        },
        DeviceInteractionRound: {
            Event.DONE: ResetAndPauseRound,
            Event.ERROR: ResetAndPauseRound
        }
    }
    final_states: Set[AppState] = set()
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: frozenset[str] = frozenset(
        [get_name(SynchronizedData.prosumer_data)]
    )
    db_pre_conditions: Dict[AppState, Set[str]] = {
        RegistrationRound: [],
    }
    db_post_conditions: Dict[AppState, Set[str]] = {

    }
