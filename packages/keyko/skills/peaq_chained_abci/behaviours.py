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

"""This package contains round behaviours of PeaqChainedSkillAbci."""

from typing import Set, Type

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.keyko.skills.peaq_abci.behaviours import (PeaqRoundBehaviour, RegistrationBehaviour)
from packages.keyko.skills.send_api_data_abci.behaviours import SendAPIDataRoundBehaviour
from packages.keyko.skills.peaq_chained_abci.composition import (
    PeaqChainedSkillAbciApp,
)

class PeaqChainedConsensusBehaviour(AbstractRoundBehaviour):
    """Class to define the behaviours this AbciApp has."""

    initial_behaviour_cls = RegistrationBehaviour
    abci_app_cls = PeaqChainedSkillAbciApp
    behaviours: Set[Type[BaseBehaviour]] = {
        *PeaqRoundBehaviour.behaviours,
        *SendAPIDataRoundBehaviour.behaviours
    }