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
    CollectDifferentUntilAllRound,
    CollectSameUntilAllRound,
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
    def prosumer_data(self) -> int:
        """Get the print count."""
        return self.db.get("prosumer_data", [])



class RegistrationRound(CollectSameUntilThresholdRound):
    """RegistrationRound"""

    payload_class = RegistrationPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData
    payload_sent = False

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""

        if not self.payload_sent:
            return None
        
        print(len(self.synchronized_data.prosumer_data))

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
    payload_attribute = ""  # TODO: update
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
        print(payload)
        prosumer_data.append(payload.prosumer_data)
        self.synchronized_data.update(
            prosumer_data=prosumer_data[-60:],
            synchronized_data_class=SynchronizedData,
        )
        self.payload_sent = True
        return


class DeviceInteractionRound(AbstractRound):
    """DeviceInteractionRound"""

    payload_class = DeviceInteractionPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    # TODO: replace AbstractRound with one of CollectDifferentUntilAllRound,
    # CollectSameUntilAllRound, CollectSameUntilThresholdRound,
    # CollectDifferentUntilThresholdRound, OnlyKeeperSendsRound, VotingRound,
    # from packages/valory/skills/abstract_round_abci/base.py
    # or implement the methods

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        raise NotImplementedError

    def check_payload(self, payload: DeviceInteractionPayload) -> None:
        """Check payload."""
        return

    def process_payload(self, payload: DeviceInteractionPayload) -> None:
        """Process payload."""
        return


class QueryModelRound(AbstractRound):
    """QueryModelRound"""

    payload_class = QueryModelPayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    # TODO: replace AbstractRound with one of CollectDifferentUntilAllRound,
    # CollectSameUntilAllRound, CollectSameUntilThresholdRound,
    # CollectDifferentUntilThresholdRound, OnlyKeeperSendsRound, VotingRound,
    # from packages/valory/skills/abstract_round_abci/base.py
    # or implement the methods

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        raise NotImplementedError

    def check_payload(self, payload: QueryModelPayload) -> None:
        """Check payload."""
        return

    def process_payload(self, payload: QueryModelPayload) -> None:
        """Process payload."""
        return


class ResetAndPauseRound(CollectSameUntilThresholdRound):
    """ResetAndPauseRound"""

    payload_class = ResetAndPausePayload
    payload_attribute = ""  # TODO: update
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Enum]]:
        """Process the end of the block."""
        if self.threshold_reached:
            return self.synchronized_data.create(), Event.DONE
        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY
        return None

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
            Event.DONE: ResetAndPauseRound
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
