from __future__ import annotations
import datetime
import hashlib
import json
import os
import pickle
import shutil
import sys
import zipfile
import pandas as pd
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
import logging
from pbpstats.client import Client
from enum import Enum

NO_PLAYER = "na"
ON_DEFENSE = "def"
ON_OFFENSE = "off"
NOT_PLAYING = "na"
PLAYER_ID_PREFIX = "pid-"
TEAM_ID_PREFIX = "t"


def player_id_from_int(player_id: int) -> str:
    return f"{PLAYER_ID_PREFIX}{player_id}"


def team_id_from_int(team_id: int) -> str:
    return f"{TEAM_ID_PREFIX}{team_id}"


def int_from_team_id(team_id: str) -> int:
    return int(team_id.lstrip(TEAM_ID_PREFIX))


def construct_year_string(year):
    return str(int(year)-1) + '-' + str(year)[2:]


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return filename


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True, default=lambda x: str(x)).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def disk_cache(cache_dir="cache"):
    """ Cache results of a function to disk based on the hash of the arguments"""
    def decorator(func):
        if type(func) == staticmethod:
            func = func.__func__

        def wrapper(*args, **kwargs):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

            dhash = dict_hash({"args": args, "kwargs": kwargs})
            filename = f"{cache_dir}/{func.__name__}_{dhash}.cache"
            if os.path.exists(filename):
                with open(filename, "rb") as fh:
                    return pickle.load(fh)
            else:
                result = func(*args, **kwargs)
                with open(filename, "wb") as fh:
                    pickle.dump(result, fh)
                return result
        return wrapper
    return decorator


@dataclass
class Team:
    name: str
    id: str


@dataclass
class Player:
    name: str
    id: str
    # team: Team
    weight_lbs: float = 0.0
    height_inches: float = 0.0
    wingspan_inches: float = 0.0
    date_of_birth: datetime.date = None

    def age_years(self, current_date: datetime.date):
        return (current_date - self.date_of_birth).total_seconds() / (3600 * 24 * 365)


@dataclass
class Possession:
    events: List[Event]


@dataclass
class Game:
    year: int
    type: str  # Regular season, play-in, playoff
    id: str
    date: datetime.date
    home_team: Team
    away_team: Team
    players: Sequence[Player]
    possessions: List[Possession]

    def __post_init__(self):
        self._player_id_to_player = {p.id: p for p in self.players}

    def get_player(self, player_id: str):
        return self._player_id_to_player[player_id]

    @property
    def events(self):
        for possession in self.possessions:
            offense_events = [e for e in possession.events if e.offense_team_id]
            if len(offense_events) == 0:
                continue

            orig_offense_team_id = offense_events[0].offense_team_id
            last_offense_event = [e for e in possession.events if e.offense_team_id == orig_offense_team_id][-1]
            for event in possession.events:
                # override lineup_ids for free throw events to correct for substitutions
                if event.type == EventType.FREE_THROW:
                    event.override_lineup_ids(last_offense_event)

                yield event


class EventType(Enum):
    START_OF_PERIOD = "StatsStartOfPeriod"
    END_OF_PERIOD = "StatsEndOfPeriod"
    JUMP_BALL = "StatsJumpBall"
    REPLAY = "StatsReplay"
    VIOLATION = "StatsViolation"
    EJECTION = "StatsEjection"
    REBOUND = "StatsRebound"
    TURNOVER = "StatsTurnover"
    FOUL = "StatsFoul"
    TIMEOUT = "StatsTimeout"
    SUBSTITUTION = "StatsSubstitution"
    FIELD_GOAL = "StatsFieldGoal"
    FREE_THROW = "StatsFreeThrow"


class Event:
    OFFENSE_EVENT_TYPES = (EventType.FIELD_GOAL, EventType.FREE_THROW, EventType.REBOUND, EventType.TURNOVER, EventType.START_OF_PERIOD, EventType.JUMP_BALL)

    def __init__(self, pbpstats_event, home_team_id: str):
        self.pbpstats_event = pbpstats_event
        delattr(self.pbpstats_event, "fouls_to_give")  # prevent pickle dump failure
        self.type = EventType(pbpstats_event.__class__.__name__)
        self._lineup_ids = {team_id_from_int(id): lineup for id, lineup in self.pbpstats_event.lineup_ids.items()}
        self._home_team_id = home_team_id

    @property
    def is_fga(self) -> bool:
        return self.type == EventType.FIELD_GOAL

    @property
    def is_fta(self) -> bool:
        return self.type == EventType.FREE_THROW

    @property
    def is_and1(self) -> bool:
        return self.is_fga and getattr(self.pbpstats_event, "is_and1", None)

    @property
    def is_heave(self) -> bool:
        return getattr(self.pbpstats_event, "is_heave", None)

    @property
    def is_foul(self):
        return self.type == EventType.FOUL

    @property
    def is_turnover(self):
        return self.type == EventType.TURNOVER

    @property
    def is_blocked(self):
        return getattr(self.pbpstats_event, "is_blocked", None)

    @property
    def is_assisted(self):
        return getattr(self.pbpstats_event, "is_assisted", None)

    @property
    def is_rebound(self):
        return self.type == EventType.REBOUND and getattr(self.pbpstats_event, "is_real_rebound", None)

    @property
    def is_ejection(self):
        return self.type == EventType.EJECTION

    @property
    def player1(self) -> Optional[str]:
        if hasattr(self.pbpstats_event, "player1_id"):
            return player_id_from_int(self.pbpstats_event.player1_id)
        else:
            return NO_PLAYER

    @property
    def player2(self) -> Optional[str]:
        if hasattr(self.pbpstats_event, "player2_id"):
            return player_id_from_int(self.pbpstats_event.player2_id)
        else:
            return NO_PLAYER

    @property
    def player3(self) -> Optional[str]:
        if hasattr(self.pbpstats_event, "player3_id"):
            return player_id_from_int(self.pbpstats_event.player3_id)
        else:
            return NO_PLAYER

    @property
    def shooter(self) -> Optional[str]:
        return self.player1 if self.is_fga else NO_PLAYER

    @property
    def assister(self) -> Optional[str]:
        return self.player2 if self.is_assisted else NO_PLAYER

    @property
    def blocker(self) -> Optional[str]:
        return self.player3 if self.is_blocked else NO_PLAYER

    @property
    def defender(self) -> Optional[str]:
        return self.player3 if self.is_fga else NO_PLAYER

    @property
    def fouler(self):
        return self.player1 if self.is_foul else NO_PLAYER

    @property
    def foulee(self):
        return self.player3 if self.is_foul else NO_PLAYER

    @property
    def stealer(self):
        return self.player3 if self.is_turnover else NO_PLAYER

    @property
    def turner_over(self):
        return self.player1 if self.is_turnover else NO_PLAYER

    @property
    def rebounder(self):
        return self.player1 if self.is_rebound else NO_PLAYER

    @property
    def ejectee(self):
        return self.player1 if self.is_ejection else NO_PLAYER

    @property
    def offense_team_id(self) -> Optional[str]:
        if hasattr(self.pbpstats_event, "team_id"):
            return team_id_from_int(self.pbpstats_event.team_id)
        else:
            return None

    @property
    def defense_team_id(self) -> Optional[str]:
        offense_team_id = self.offense_team_id
        if offense_team_id:
            team_ids = set(map(team_id_from_int, self.pbpstats_event.current_players.keys()))
            team_ids.remove(offense_team_id)
            return team_ids.pop()
        else:
            return None

    @property
    def offense_is_home(self) -> bool:
        return self.offense_team_id == self._home_team_id

    @property
    def shot_distance(self):
        return getattr(self.pbpstats_event, "distance", None)

    @property
    def shot_value(self):
        return getattr(self.pbpstats_event, "shot_value", None)

    @property
    def is_made(self) -> Optional[bool]:
        return getattr(self.pbpstats_event, "is_made", None)

    def override_lineup_ids(self, event: Event):
        self._lineup_ids = event._lineup_ids

    @property
    def offense_player_ids(self) -> Sequence[str]:
        return list(map(player_id_from_int, self._lineup_ids[self.offense_team_id].split("-")))

    @property
    def defense_player_ids(self) -> Sequence[str]:
        return list(map(player_id_from_int, self._lineup_ids[self.defense_team_id].split("-")))

    @property
    def offense_score(self) -> int:
        return self.pbpstats_event.score[int_from_team_id(self.offense_team_id)]

    @property
    def defense_score(self) -> int:
        return self.pbpstats_event.score[int_from_team_id(self.defense_team_id)]

    @property
    def seconds_remaining(self) -> float:
        return self.pbpstats_event.seconds_remaining


class Dataset:
    """ Multiple NBA games """
    @staticmethod
    def download_pbpstats_data(pbpstats_data_fn, out_data_dir):
        if not os.path.exists(out_data_dir):
            download_file("https://pbpstats.s3.amazonaws.com/data.zip", pbpstats_data_fn)
            with zipfile.ZipFile(pbpstats_data_fn, 'r') as zip_ref:
                zip_ref.extractall(out_data_dir)

    @disk_cache()
    @staticmethod
    def load_from_disk(data_dir: str, years: Sequence[int], season_types: List[str], max_games: Optional[int] = None):
        Dataset.download_pbpstats_data("pbpstats_data.zip", data_dir)

        games = []
        all_player_id_to_player = {}
        for year in years:
            for season_type in season_types:
                player_id_to_player = {}
                year_string = construct_year_string(year)
                logging.info(f"Loading data for season {year_string} {season_type}")
                settings = {
                    "dir": f"{data_dir}/response_data",
                    "Boxscore": {"source": "file", "data_provider": "stats_nba"},  # game.boxscore.items
                    "Games": {"source": "file", "data_provider": "stats_nba"},
                    "Possessions": {"source": "file", "data_provider": "stats_nba"},  # game.possessions.items
                }
                client = Client(settings)
                season_api = client.Season("nba", year_string, season_type)
                pbpstats_games = season_api.games.final_games
                if max_games:
                    pbpstats_games = pbpstats_games[:max_games]

                for pbpstats_game in pbpstats_games:
                    logging.debug(f"Loading game: {pbpstats_game['game_id']}")
                    full_pbpstats_game = client.Game(pbpstats_game["game_id"])
                    for player_id, player_name in full_pbpstats_game.boxscore.player_name_map.items():
                        player_id_to_player[player_id_from_int(player_id)] = Player(player_name, player_id_from_int(player_id))

                    game_id = pbpstats_game["game_id"]
                    date = pbpstats_game["date"]

                    teams = [Team(team_item["team_abbreviation"], team_id_from_int(team_item["team_id"]))
                             for team_item in full_pbpstats_game.boxscore.team_items]
                    if teams[0].id == pbpstats_game['home_team_id']:
                        home_team, away_team = teams[0], teams[1]
                    else:
                        home_team, away_team = teams[1], teams[0]

                    possessions = []
                    for pbp_possession in full_pbpstats_game.possessions.items:
                        possessions.append(Possession([Event(e, home_team.id) for e in pbp_possession.events]))

                    games.append(Game(year, season_type, game_id, date, home_team, away_team,
                                      list(player_id_to_player.values()), possessions))
                    all_player_id_to_player.update(player_id_to_player)

        return Dataset(games, all_player_id_to_player)

    def __init__(self, games: Sequence[Game], player_id_to_player: Dict[str, Player]):
        self._games = games
        self._player_id_to_player = player_id_to_player

    @property
    def games(self):
        return self._games

    @property
    def player_id_to_player(self):
        return self._player_id_to_player


class DatasetToTable:
    SKIP_EVENT_TYPES = (EventType.START_OF_PERIOD, EventType.JUMP_BALL,
                        EventType.TIMEOUT, EventType.END_OF_PERIOD, EventType.REPLAY)

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._player_id_to_name = {pid: p.name for pid, p in self._dataset.player_id_to_player.items()}
        # self._player_name_to_id = {name: id for id, name in self._player_id_to_name.items()}
        self._player_default_row_data = pd.Series({pid: NOT_PLAYING for pid in self._player_id_to_name.keys()})

    def _event_to_row_data(self, event: Event) -> pd.Series:
        player_row_data = self._player_default_row_data.copy()
        player_row_data[event.offense_player_ids] = ON_OFFENSE
        player_row_data[event.defense_player_ids] = ON_DEFENSE
        data = {
             "offense_is_home": event.offense_is_home,
             "offense_score": event.offense_score,
             "defense_score": event.defense_score,
             "seconds_remaining": event.seconds_remaining,
             # "clock_secs_left": game_state.clock_secs_left,
             "fg": event.is_fga,
             "fg:shooter": event.shooter,
             "fg:assister": event.assister,
             "fg:defender":event.defender,
             # "fg_defender_distance": event.defender_distance,
             "fg:blocker": event.blocker,
             "fg:is_3pa": event.shot_value == 3,
             "fg:distance": event.shot_distance,
             "fg:is_made": event.is_made,
             # "foul": event.is_foul,
             # "foul:fouler": event.fouler,
             # "foul:foulee": event.foulee,
             # "to": event.is_turnover,
             # "to:turner_over": event.turner_over,
             # "to:stealer": event.stealer,
             # "ft": event.is_fta,
             # "ft:is_made": event.is_made,
             # "rebound": event.is_rebound,
             # "rebound:rebounder": event.rebounder,
             # "ejection": event.is_ejection,
             # "ejection:ejectee": event.ejectee
            }
        return pd.concat([player_row_data, pd.Series(data)])

    @property
    def df(self) -> pd.DataFrame:
        logging.info(f"Converting Dataset to table")

        rows = []
        for game in self._dataset.games:
            for event in game.events:
                if event.type not in self.SKIP_EVENT_TYPES and event.player1:
                    if event.type == EventType.FIELD_GOAL:  # TODO
                        rows.append(self._event_to_row_data(event))
        df = pd.concat(rows, ignore_index=True, axis=1).T
        for score_col in [col for col in df.columns if "score" in col]:
            df[score_col] = df[score_col].astype(np.float32)

        # validate
        total_players = len(self._player_default_row_data)
        players_df = df[df.columns[:total_players]]
        num_players_per_row = (players_df != NOT_PLAYING).sum(axis=1)
        assert (num_players_per_row == 10).all()
        assert (df.offense_score >= 0).all()
        assert (df.defense_score >= 0).all()

        return df


def main():
    logging.basicConfig(level=logging.DEBUG)

    data_dir = sys.argv[1]
    start_year = int(sys.argv[2])
    end_year = int(sys.argv[3])
    max_games = int(sys.argv[4])

    dataset = Dataset.load_from_disk(data_dir, range(start_year, end_year+1), ["Regular Season"], max_games)
    df = DatasetToTable(dataset).df
    df.to_csv(f"nba_pbp_{start_year}-{end_year}_n{max_games}.csv", index=False)


if __name__ == "__main__":
    main()
