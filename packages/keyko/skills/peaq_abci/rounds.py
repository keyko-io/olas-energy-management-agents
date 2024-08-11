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
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AbstractRound,
    AppState,
    BaseSynchronizedData,
    DegenerateRound,
    EventToTimeout,
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
    RESET_TIMEOUT = "reset_timeout"
    NOT_ENOUGH_DATA = "not_enough_data"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """


class CollectDataRound(AbstractRound):
    """CollectDataRound"""

    payload_class = CollectDataPayload
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

    def check_payload(self, payload: CollectDataPayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: CollectDataPayload) -> None:
        """Process payload."""
        raise NotImplementedError


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
        raise NotImplementedError

    def process_payload(self, payload: DeviceInteractionPayload) -> None:
        """Process payload."""
        raise NotImplementedError


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
        raise NotImplementedError

    def process_payload(self, payload: QueryModelPayload) -> None:
        """Process payload."""
        raise NotImplementedError


class RegistrationRound(AbstractRound):
    """RegistrationRound"""

    payload_class = RegistrationPayload
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

    def check_payload(self, payload: RegistrationPayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: RegistrationPayload) -> None:
        """Process payload."""
        raise NotImplementedError


class ResetAndPauseRound(AbstractRound):
    """ResetAndPauseRound"""

    payload_class = ResetAndPausePayload
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

    def check_payload(self, payload: ResetAndPausePayload) -> None:
        """Check payload."""
        raise NotImplementedError

    def process_payload(self, payload: ResetAndPausePayload) -> None:
        """Process payload."""
        raise NotImplementedError


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
            Event.TRANSACT: DeviceInteractionRound
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
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        RegistrationRound: [],
    }
    db_post_conditions: Dict[AppState, Set[str]] = {

    }
