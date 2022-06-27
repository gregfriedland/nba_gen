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

NO_TEAM = "na"
NO_PLAYER = "na"
ON_DEFENSE = "def"
ON_OFFENSE = "off"
NOT_PLAYING = "na"
PLAYER_ID_PREFIX = "pid-"
TEAM_ID_PREFIX = "t"


def player_id_from_int(player_id: int) -> str:
    if player_id == 0:
        return NO_PLAYER
    else:
        return f"{PLAYER_ID_PREFIX}{player_id}"


def team_id_from_int(team_id: int) -> str:
    if team_id < 10000:
        return NO_TEAM
    else:
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

    @property
    def cleaned_events(self):
        offense_events = [e for e in self.events if e.type in Event.OFFENSE_EVENT_TYPES and
                          not (e.is_technical_foul or e.is_technical_fta or e.is_double_foul)]
        if len(offense_events) == 0:
            return None
        # elif len(offense_events) == 1 and offense_events[0].type == EventType.END_OF_PERIOD:
        #     return None
        # if event.type == EventType.FOUL and event.offense_team_id == NO_TEAM:

        orig_offense_team_id = offense_events[0].offense_team_id
        last_offense_event = [e for e in self.events if e.offense_team_id == orig_offense_team_id][-1]
        for event in self.events:
            # override lineup_ids for free throw events to correct for substitutions
            if event.type == EventType.FREE_THROW:
                event.override_lineup_ids(last_offense_event)

        return offense_events


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

    # @property
    # def cleaned_possessions(self):
    #     for possession in self.possessions:
    #         offense_events = [e for e in possession.events if e.type in Event.OFFENSE_EVENT_TYPES]
    #         if len(offense_events) == 0:
    #             continue
    #         elif len(offense_events) == 1 and offense_events[0].type == EventType.END_OF_PERIOD:
    #             continue
    #         # if event.type == EventType.FOUL and event.offense_team_id == NO_TEAM:
    #
    #         orig_offense_team_id = offense_events[0].offense_team_id
    #         last_offense_event = [e for e in possession.events if e.offense_team_id == orig_offense_team_id][-1]
    #         for event in possession.events:
    #             # override lineup_ids for free throw events to correct for substitutions
    #             if event.type == EventType.FREE_THROW:
    #                 event.override_lineup_ids(last_offense_event)
    #
    #         yield Possession(offense_events)


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
    OFFENSE_EVENT_TYPES = (EventType.FIELD_GOAL, EventType.FREE_THROW, EventType.REBOUND, EventType.TURNOVER,
                           EventType.END_OF_PERIOD, EventType.FOUL)

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
    def is_technical_foul(self) -> bool:
        return self.type == EventType.FOUL and getattr(self.pbpstats_event, "is_technical", False)

    @property
    def is_double_foul(self) -> bool:
        return self.type == EventType.FOUL and getattr(self.pbpstats_event, "is_double_foul", False)

    @property
    def is_technical_fta(self) -> bool:
        return self.type == EventType.FREE_THROW and getattr(self.pbpstats_event, "is_technical_ft", False)

    @property
    def is_and1(self) -> bool:
        return self.is_fga and getattr(self.pbpstats_event, "is_and1", False)

    @property
    def is_heave(self) -> bool:
        return getattr(self.pbpstats_event, "is_heave", False)

    @property
    def is_foul(self):
        return self.type == EventType.FOUL

    @property
    def is_turnover(self):
        return self.type == EventType.TURNOVER

    @property
    def is_blocked(self):
        return getattr(self.pbpstats_event, "is_blocked", False)

    @property
    def is_assisted(self):
        return getattr(self.pbpstats_event, "is_assisted", False)

    @property
    def is_rebound(self):
        return self.type == EventType.REBOUND and getattr(self.pbpstats_event, "is_real_rebound", False)

    @property
    def is_ejection(self):
        return self.type == EventType.EJECTION

    @property
    def is_offensive_rebound(self) -> bool:
        return getattr(self.pbpstats_event, "oreb", False)

    @property
    def is_offensive_foul(self) -> bool:
        return getattr(self.pbpstats_event, "is_charge", False) or getattr(self.pbpstats_event, "is_offensive_foul", False)

    @property
    def is_defensive_foul(self) -> bool:
        return self.is_foul and not self.is_offensive_foul

    @property
    def is_end_of_period(self) -> bool:
        return self.type == EventType.END_OF_PERIOD

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
    def other_team_id(self) -> str:
        if self.team_id == NO_TEAM:
            return NO_TEAM

        team_ids = set(map(team_id_from_int, self.pbpstats_event.current_players.keys()))
        team_ids.remove(self.team_id)
        assert len(team_ids) == 1
        return team_ids.pop()

    @property
    def team_id(self) -> str:
        if hasattr(self.pbpstats_event, "team_id"):
            return team_id_from_int(self.pbpstats_event.team_id)
        else:
            return NO_TEAM

    @property
    def offense_team_id(self) -> str:
        if self.is_defensive_foul:
            return self.team_id
        else:
            return self.other_team_id

    @property
    def defense_team_id(self) -> Optional[str]:
        if self.is_defensive_foul:
            return self.other_team_id
        else:
            return self.team_id

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
        return getattr(self.pbpstats_event, "is_made", False)

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
    def secs_left(self) -> float:
        return self.pbpstats_event.seconds_remaining

    @property
    def period(self) -> int:
        return self.pbpstats_event.period

    def __repr__(self):
        return f"{self.type}: offense={self.offense_team_id}"

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
        all_team_id_to_team = {}
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
                    team_id_to_team = {t.id: t for t in teams}
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
                    all_team_id_to_team.update(team_id_to_team)

        return Dataset(games, all_team_id_to_team, all_player_id_to_player)

    def __init__(self, games: Sequence[Game], team_id_to_team: Dict[str, Team],
                 player_id_to_player: Dict[str, Player]):
        self.games = games
        self.player_id_to_player = player_id_to_player
        self.team_id_to_team = team_id_to_team


class DatasetToTable:
    """
    Convert Dataset to table representation with format:
    ORL,NOP,...,stat.offense_is_home,stat.offense_score,stat.defense_score,stat.seconds_left,stat.fg,stat.fg:is_assisted,stat.fg:is_blocked,stat.fg:is_3pa,stat.fg:distance,stat.fg:is_made
    """
    SKIP_EVENT_TYPES = (EventType.START_OF_PERIOD, EventType.JUMP_BALL,
                        EventType.TIMEOUT, EventType.END_OF_PERIOD, EventType.REPLAY)

    def __init__(self, dataset: Dataset, teams_only: bool):
        self._dataset = dataset
        self._teams_only = teams_only
        if teams_only:
            self._default_row_data = pd.Series({t.name: NOT_PLAYING for t in self._dataset.team_id_to_team.values()})
        else:
            self._default_row_data = pd.Series({pid: NOT_PLAYING for pid in self._dataset.player_id_to_player.keys()})

    def _event_to_row_data(self, event: Event) -> pd.Series:
        row_data = self._default_row_data.copy()
        if self._teams_only:
            row_data[self._dataset.team_id_to_team[event.offense_team_id].name] = ON_OFFENSE
            row_data[self._dataset.team_id_to_team[event.defense_team_id].name] = ON_DEFENSE
        else:
            row_data[event.offense_player_ids] = ON_DEFENSE
            row_data[event.defense_player_ids] = ON_DEFENSE

        data = {
             "stat.offense_is_home": event.offense_is_home,
             "stat.offense_score": event.offense_score,
             "stat.defense_score": event.defense_score,
             "stat.seconds_left": event.seconds_remaining,
             # "stat.clock_secs_left": game_state.clock_secs_left,
             "stat.fg": event.is_fga,
             "stat.fg:is_assisted": event.is_assisted,
             "stat.fg:is_blocked": event.is_blocked,
             # "stat.fg:shooter": event.shooter,
             # "stat.fg:assister": event.assister,
             # "stat.fg:defender":event.defender,
             # "stat.fg_defender_distance": event.defender_distance,
             # "stat.fg:blocker": event.blocker,
             "stat.fg:is_3pa": event.shot_value == 3,
             "stat.fg:distance": event.shot_distance,
             "stat.fg:is_made": event.is_made,
             # "stat.foul": event.is_foul,
             # "stat.foul:fouler": event.fouler,
             # "stat.foul:foulee": event.foulee,
             # "stat.to": event.is_turnover,
             # "stat.to:turner_over": event.turner_over,
             # "stat.to:stealer": event.stealer,
             # "stat.ft": event.is_fta,
             # "stat.ft:is_made": event.is_made,
             # "stat.rebound": event.is_rebound,
             # "stat.rebound:rebounder": event.rebounder,
             # "stat.ejection": event.is_ejection,
             # "stat.ejection:ejectee": event.ejectee
            }
        return pd.concat([row_data, pd.Series(data)])

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
        num_entities = len(self._default_row_data)
        entity_df = df[df.columns[:num_entities]]
        num_entities_per_row = (entity_df != NOT_PLAYING).sum(axis=1)
        if self._teams_only:
            assert num_entities == 30
            assert (num_entities_per_row == 2).all()
        else:
            assert (num_entities_per_row == 10).all()
        assert (df["stat.offense_score"] >= 0).all()
        assert (df["stat.defense_score"] >= 0).all()

        return df


class DatasetToTable2:
    """
    Convert Dataset to table with format:
    Off team, Def team, Off is home?, Quarter num, Quarter time left, Num off rebounds, Num shots blocked, Num def fouls,
        Num FTAs, Was shot made?, Was turnover stolen?, Was fg assisted?, End type,	Shot clock secs left, Shot distance, Shot is 3pa?
    End type = FGM, Def rebound, turnover, offensive foul, end of quarter (skip)

    """
    @staticmethod
    def _get_single_event_value(events: List[Event], field: str):
        unique_vals = set(getattr(e, field) for e in events)
        if len(unique_vals) != 1:
            raise ValueError(f"Expected exactly 1 value of '{field}' from events but got: {unique_vals}")
        return unique_vals.pop()

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._default_row_data = pd.Series({t.name: NOT_PLAYING for t in self._dataset.team_id_to_team.values()})

    def _events_to_row_data(self, events: List[Event]) -> pd.Series:
        row_data = self._default_row_data.copy()
        off_team_id = events[0].offense_team_id
        def_team_id = events[0].defense_team_id
        off_is_home = events[0].offense_is_home

        num_off_rebounds = len([e for e in events if e.is_offensive_rebound])
        num_shots_blocked = len([e for e in events if e.is_blocked])
        num_def_fouls = len([e for e in events if e.is_defensive_foul])
        num_ftas = len([e for e in events if e.is_fta])
        num_ftms = len([e for e in events if e.is_fta and e.is_made])

        row_data[self._dataset.team_id_to_team[off_team_id].name] = ON_OFFENSE
        row_data[self._dataset.team_id_to_team[def_team_id].name] = ON_DEFENSE

        if events[-1].is_offensive_rebound:
            logging.warning(f"Unexpected offensive rebound as last event")
        if events[-1].is_defensive_foul:
            logging.warning(f"Unexpected defensive foul as last event")
        if events[0].secs_left > 12 * 60:
            logging.warning(f"Unexpected secs left > 12min")

        data = {
            "stat.offense_is_home": off_is_home,
            "stat.period": events[0].period,
            "stat.period_secs_left": events[0].secs_left,
            "stat.num_off_rebounds": num_off_rebounds,
            "stat.num_shots_blocked": num_shots_blocked,
            "stat.num_def_fouls": num_def_fouls,
            "stat.num_ftas": num_ftas,
            "stat.num_ftms": num_ftms,

            "stat.fg_is_made": events[-1].is_made,
            "stat.fg_is_assisted": events[-1].is_assisted,
            "stat.fg_is_3pa": events[-1].shot_value == 3,
            "stat.fg_distance": events[-1].shot_distance or 0,
            "stat.to_is_stolen": events[-1].stealer != NO_PLAYER,

            "stat.end_is_foul": events[-1].is_foul,
            "stat.end_is_fga": events[-1].is_fga,
            "stat.end_is_to": events[-1].is_turnover,
            "stat.end_is_blocked": events[-1].is_blocked,
            "stat.end_is_def_rebound": events[-1].is_rebound,
            "stat.end_is_period_over": events[-1].is_end_of_period,

            "stat.off_score_margin": events[0].offense_score - events[0].defense_score,
        }
        return pd.concat([row_data, pd.Series(data)])

    @property
    def df(self) -> pd.DataFrame:
        logging.info(f"Converting Dataset to table")

        rows = []
        for game in self._dataset.games:
            last_possession = None
            for possession in game.possessions:
                cleaned_events = possession.cleaned_events
                if cleaned_events is None:
                    continue
                elif set(e.type for e in cleaned_events) == {EventType.END_OF_PERIOD}:
                    last_possession = None
                    continue

                logging.debug(possession)
                rows.append(self._events_to_row_data(cleaned_events))

                # sanity check that possessions alternate offense teams
                if last_possession and last_possession.cleaned_events[0].offense_team_id == cleaned_events[0].offense_team_id:
                    logging.warning(f"Offense team did not change between possessions")
                if cleaned_events[-1].type == EventType.END_OF_PERIOD:
                    last_possession = None
                else:
                    last_possession = possession
        df = pd.concat(rows, ignore_index=True, axis=1).T
        for score_col in [col for col in df.columns if "score" in col]:
            df[score_col] = df[score_col].astype(np.float32)

        # validate
        num_entities = len(self._default_row_data)
        entity_df = df[df.columns[:num_entities]]
        num_entities_per_row = (entity_df != NOT_PLAYING).sum(axis=1)
        assert num_entities == 30
        assert (num_entities_per_row == 2).all()

        return df


def main():
    logging.basicConfig(level=logging.INFO)

    data_dir = sys.argv[1]
    start_year = int(sys.argv[2])
    end_year = int(sys.argv[3])
    max_games = int(sys.argv[4])
    # teams_only = sys.argv[5].lower() == "teams"

    dataset = Dataset.load_from_disk(data_dir, range(start_year, end_year+1), ["Regular Season"], max_games)
    df = DatasetToTable2(dataset).df
    df.to_csv(f"nba_pbp_{start_year}-{end_year}_n{max_games}.csv", index=False)


if __name__ == "__main__":
    main()
