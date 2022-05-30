from __future__ import annotations
import datetime
import os
import shutil
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional

import requests
from utils import construct_year_string, disk_cache
import logging
from pbpstats.client import Client
from enum import Enum


@dataclass
class Team:
    name: str
    abbreviation: str
    id: int


@dataclass
class Player:
    name: str
    id: int
    team: Team
    weight_lbs: float
    height_inches: float
    wingspan_inches: float
    date_of_birth: datetime.date
    # acquired: List[datetime.date, Team]

    def age_years(self, current_date: datetime.date):
        return (current_date - self.date_of_birth).total_seconds() / (3600 * 24 * 365)


@dataclass
class Game:
    year: int
    type: str  # Regular season, play-in, playoff
    id: str
    date: datetime.date
    home_team: Team
    away_team: Team
    players: List[Player]
    pbpstats_possessions: List

    def __post_init__(self):
        self._player_id_to_player = {p.id: p for p in self.players}

    def get_player(self, player_id: int):
        return self._player_id_to_player[player_id]

    @property
    def events(self):
        for pbpstats_possession in self.pbpstats_possessions:
            events = [Event(e) for e in pbpstats_possession.events]

            orig_offense_team_id = [e.offense_team_id for e in events if e.offense_team_id][0]
            last_offense_event = [e for e in events if e.offense_team_id == orig_offense_team_id][-1]
            for event in events:
                # override lineup_ids for free throw events to correct for substitutions
                if event.type == EventType.FREE_THROW:
                    event.override_lineup_ids(last_offense_event)

                yield event
