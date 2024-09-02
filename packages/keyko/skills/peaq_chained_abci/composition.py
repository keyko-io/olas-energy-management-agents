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

"""This package contains round behaviours of PeaqChainedSkillAbciApp."""

import packages.keyko.skills.peaq_abci.rounds as PeaqAbci
import packages.keyko.skills.send_api_data_abci.rounds as SendAPIDataAbci
from packages.valory.skills.abstract_round_abci.abci_app_chain import (
    AbciAppTransitionMapping,
    chain,
)

abci_app_transition_mapping: AbciAppTransitionMapping = {
    PeaqAbci.FinishedRound: SendAPIDataAbci.ProjectDataSubmissionDecisionRound,
    SendAPIDataAbci.FinishedAgentDataSubmissionRound: PeaqAbci.CollectDataRound,
}

PeaqChainedSkillAbciApp = chain(
    (
        PeaqAbci.PeaqAbciApp,
        SendAPIDataAbci.SendAPIDataAbciApp,
    ),
    abci_app_transition_mapping,
)
