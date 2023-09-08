from typing import Optional

import requests
from urllib.parse import urljoin
import time
import json
import os
import getpass
from pathlib import Path

import pandas as pd

from multiprocessing.pool import ThreadPool
from functools import partial

import tqdm
from helpful_functions import save_simulation_result, set_alpha_properties, save_pnl, save_yearly_stats, get_alpha_pnl, get_alpha_yearly_stats

DEFAULT_CONFIG = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False,
    "save_stats_file": False,
    "save_result_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}


def get_credentials():
    """
    Function gets json with platform credentials if exists,
    or asks to enter new ones
    """

    credential_email = os.environ.get('ACE_CREDENTIAL_EMAIL')
    credential_password = os.environ.get('ACE_CREDENTIAL_PASSWORD')

    credentials_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
    credentials_file_path = os.path.join(credentials_folder_path, "platform-brain.json")

    if (
        Path(credentials_file_path).exists()
        and os.path.getsize(credentials_file_path) > 2
    ):
        with open(credentials_file_path) as file:
            data = json.loads(file.read())
    else:
        os.makedirs(credentials_folder_path, exist_ok=True)
        if credential_email and credential_password:
            email = credential_email
            password = credential_password
        else:
            email = input("Email:\n")
            password = getpass.getpass(prompt="Password:")
        data = {"email": email, "password": password}
        with open(credentials_file_path, "w") as file:
            json.dump(data, file)
    return (data["email"], data["password"])


def start_session(): 

    """
    Function sign in to platform
    and checks credentials
    and returns session object
    """

    s = requests.Session()
    s.auth = get_credentials()
    r = s.post("https://api.worldquantbrain.com/authentication")

    if r.status_code == requests.status_codes.codes.unauthorized:
        if r.headers["WWW-Authenticate"] == "persona":
            print(
                "Complete biometrics authentication and press any key to continue: \n"
                + urljoin(r.url, r.headers["Location"]) + "\n"
            )
            input()
            s.post(urljoin(r.url, r.headers["Location"]))
            
            while True:
                if s.post(urljoin(r.url, r.headers["Location"])).status_code != 201:
                    input("Biometrics authentication is not complete. Please try again and press any key when completed \n")
                else:
                    break
        else:
            print("\nIncorrect email or password\n")
            with open(
                os.path.join(os.path.expanduser("~"), "secrets/platform-brain.json"),
                "w",
            ) as file:
                json.dump({}, file)
            return start_session()
    return s

def check_session_timeout(s):
    """
    Function checks session time out
    """

    authentication_url = "https://api.worldquantbrain.com/authentication"
    try:
        result = s.get(authentication_url).json()["token"]["expiry"]
        return result
    except:
        return 0


def generate_alpha(
    regular: str,
    region: str = "USA",
    universe: str = "TOP3000",
    neutralization: str = "INDUSTRY",
    delay: int = 1,
    decay: int = 0,
    truncation: float = 0.08,
    nan_handling: str = "OFF",
    unit_handling: str = "VERIFY",
    pasteurization: str = "ON",
    visualization: bool = False,
):
    """
    Function generates data to use in simulation
    has default parameters
    """

    simulation_data = {
        "type": "REGULAR",
        "settings": {
            "nanHandling": nan_handling,
            "instrumentType": "EQUITY",
            "delay": delay,
            "universe": universe,
            "truncation": truncation,
            "unitHandling": unit_handling,
            "pasteurization": pasteurization,
            "region": region,
            "language": "FASTEXPR",
            "decay": decay,
            "neutralization": neutralization,
            "visualization": visualization,
        },
        "regular": regular,
    }
    return simulation_data


def start_simulation(
    s, simulate_data
):  
    simulate_response = s.post(
        "https://api.worldquantbrain.com/simulations", json=simulate_data
    )
    return simulate_response


def simulation_progress(s,
    simulate_response,
):  

    if simulate_response.status_code // 100 != 2:
        print(simulate_response.text)
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        simulation_progress = s.get(simulation_progress_url)
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:

        print("An error occurred")
        if "message" in simulation_progress.json():
            print(simulation_progress.json()["message"])
        return {"completed": False, "result": {}}

    alpha = simulation_progress.json().get("alpha", 0)
    if alpha == 0:
        return {"completed": False, "result": {}}
    simulation_result = get_simulation_result_json(s, alpha)
    return {"completed": True, "result": simulation_result}



def multisimulation_progress(s,
    simulate_response,
): 
    
    if simulate_response.status_code // 100 != 2:
        print(simulate_response.text)
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        simulation_progress = s.get(simulation_progress_url)
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:
        print("An error occurred")
        if "message" in simulation_progress.json():
            print(simulation_progress.json()["message"])
        return {"completed": False, "result": {}}


    children = simulation_progress.json().get("children", 0)
    if len(children) == 0:
        return {"completed": False, "result": {}}
    children_list = []
    for child in children:
        child_progress = s.get("https://api.worldquantbrain.com/simulations/" + child)
        alpha = child_progress.json()["alpha"]
        child_result = get_simulation_result_json(s, alpha)
        children_list.append(child_result)
    return {"completed": True, "result": children_list}


def get_prod_corr(s, alpha_id):
    """
    Function gets alpha's prod correlation
    and save result to dataframe
    """

    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/correlations/prod"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    prod_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)

    return prod_corr_df


def check_prod_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's prod_corr test passed
    Saves result to dataframe
    """

    prod_corr_df = get_prod_corr(s, alpha_id)
    value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max()
    result = [
        {"test": "PROD_CORRELATION", "result": "PASS" if value <= threshold else "FAIL", "limit": threshold, "value": value, "alpha_id": alpha_id}
    ]
    return pd.DataFrame(result)


def get_self_corr(s, alpha_id):
    """
    Function gets alpha's self correlation
    and save result to dataframe
    """

    while True:

        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/correlations/self"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    
    records_len = len(result.json()["records"])
    if records_len == 0:
        return pd.DataFrame()

    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)

    return self_corr_df


def check_self_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's self_corr test passed
    Saves result to dataframe
    """

    self_corr_df = get_self_corr(s, alpha_id)
    if self_corr_df.empty:
        result = [{"test": "SELF_CORRELATION", "result": "PASS", "limit": threshold, "value": 0, "alpha_id": alpha_id}]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id
            }
        ]
    return pd.DataFrame(result)



def get_check_submission(s, alpha_id):
    """
    Function gets alpha's check submission checks
    and returns result in dataframe
    """

    while True:
        result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/check")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("is", 0) == 0:
        return pd.DataFrame()
    
    checks_df = pd.DataFrame(
            result.json()["is"]["checks"]
    ).assign(alpha_id=alpha_id)
    
    if 'year' in checks_df:
        ladder_dict = [checks_df.loc[checks_df.index[checks_df['name']=='IS_LADDER_SHARPE']][['value', 'year']].iloc[0].to_dict()]
        checks_df.at[checks_df.index[checks_df['name']=='IS_LADDER_SHARPE'], 'value'] = ladder_dict
        checks_df.drop(['endDate', 'startDate', 'year'], axis=1, inplace=True)

    return checks_df


def performance_comparison(s, alpha_id, team_id:Optional[str] = None, competition:Optional[str] = 'ACE2023'):
    """
    Returns performance comparison for merged performance check
    """
    if competition is not None:
        part_url = f'competitions/{competition}'
    elif team_id is not None:
        part_url = f'teams/{team_id}'
    else:
        part_url = 'users/self'
    while True:
        result = s.get(
            f"https://api.worldquantbrain.com/{part_url}/alphas/" + alpha_id + "/before-and-after-performance"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("stats", 0) == 0:
        return {}
    if result.status_code != 200:
        return {}

    return result.json()


def submit_alpha(s, alpha_id):
    """
    Function submits an alpha
    This function is not used anywhere
    """
    result = s.post("https://api.worldquantbrain.com/alphas/" + alpha_id + "/submit")
    while True:
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
            result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/submit")
        else:
            break
    return result.status_code == 200


def get_simulation_result_json(s, alpha_id):
    return s.get("https://api.worldquantbrain.com/alphas/" + alpha_id).json()


def simulate_single_alpha(
    s,
    simulate_data,
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()

    simulate_response = start_simulation(s, simulate_data)
    simulation_result = simulation_progress(s, simulate_response)
    
    if not simulation_result["completed"]:
        return {'alpha_id': None, 'simulate_data': simulate_data}
    set_alpha_properties(s, simulation_result["result"]["id"])
    return {'alpha_id': simulation_result["result"]["id"], 'simulate_data': simulate_data}


def simulate_multi_alpha(
    s,
    simulate_data_list,
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()
    if len(simulate_data_list) == 1:
        return [simulate_single_alpha(s, simulate_data_list[0])]
    simulate_response = start_simulation(s, simulate_data_list)
    simulation_result = multisimulation_progress(s, simulate_response)
    
    if not simulation_result["completed"]:
        return [{'alpha_id': None, 'simulate_data': x} for x in simulate_data_list]
    result = [{"alpha_id": x["id"], "simulate_data": {"type": x["type"], "settings": x["settings"], "regular": x["regular"]["code"]}} for x in simulation_result["result"]]
    _ = [set_alpha_properties(s, x["id"]) for x in simulation_result["result"]]
    return result
    

def get_specified_alpha_stats(
    s,
    alpha_id,
    simulate_data,
    get_pnl: bool = False,
    get_stats: bool = False,
    save_pnl_file: bool = False,
    save_stats_file: bool = False,
    save_result_file: bool = False,
    check_submission: bool = False,
    check_self_corr: bool = False,
    check_prod_corr: bool = False,
):
    """
    Master-Function to get specified in config statistics

    """
    pnl = None
    stats = None

    if alpha_id is None:
        return {'alpha_id': None, 'simulate_data': simulate_data, 'is_stats': None, 'pnl': pnl, 'stats': stats, 'is_tests': None}

    result = get_simulation_result_json(s, alpha_id)
    region = result["settings"]["region"]
    is_stats = pd.DataFrame([{key: value for key, value in result['is'].items() if key!='checks'}]).assign(alpha_id=alpha_id)
    
    if get_pnl:
        pnl = get_alpha_pnl(s, alpha_id)
    if get_stats:
        stats = get_alpha_yearly_stats(s, alpha_id)
    
    if save_result_file:
        save_simulation_result(result)
    if save_pnl_file and get_pnl:
        save_pnl(pnl, alpha_id, region)
    if save_stats_file and get_stats:
        save_yearly_stats(stats, alpha_id, region)

    is_tests = pd.DataFrame(
        result["is"]["checks"]
    ).assign(alpha_id=alpha_id)

    if check_submission:
        is_tests = get_check_submission(s, alpha_id)

        return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests}

    if check_self_corr and not check_submission:
        self_corr_test = check_self_corr_test(s, alpha_id)
        is_tests = (
            is_tests.append(self_corr_test, ignore_index=True, sort=False)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )
    if check_prod_corr and not check_submission:
        prod_corr_test = check_prod_corr_test(s, alpha_id)
        is_tests = (
            is_tests.append(prod_corr_test, ignore_index=True, sort=False)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )

    return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests}


def simulate_alpha_list(
    s,
    alpha_list,
    limit_of_concurrent_simulations=3,
    simulation_config=DEFAULT_CONFIG,
):
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(alpha_list)) as pbar:
            
            for result in pool.imap_unordered(
                partial(simulate_single_alpha, s), alpha_list
            ):
                result_list.append(result)
                pbar.update()

    stats_list_result = []
    func = lambda x: get_specified_alpha_stats(s, x['alpha_id'], x['simulate_data'], **simulation_config)
    with ThreadPool(3) as pool:
        for result in pool.map(
            func, result_list
        ):
            stats_list_result.append(result)
    
    return stats_list_result 


def simulate_alpha_list_multi(
    s,
    alpha_list,
    limit_of_concurrent_simulations=3,
    limit_of_multi_simulations=3,
    simulation_config=DEFAULT_CONFIG,
):
    if (limit_of_multi_simulations<2) or (limit_of_multi_simulations>10):
        print('Warning, limit of multi-simulation should be 2..10')
        limit_of_multi_simulations = 3
    if len(alpha_list)<10:
        print('Warning, list of alphas too short, single concurrent simulations will be used instead of multisimulations')
        return simulate_alpha_list(s, alpha_list, simulation_config=simulation_config)
    
    tasks = [alpha_list[i:i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)]
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(tasks)) as pbar:
                
            for result in pool.imap_unordered(
                partial(simulate_multi_alpha, s), tasks
            ):
                result_list.append(result)
                pbar.update()
    result_list_flat = [item for sublist in result_list for item in sublist]
    
    stats_list_result = []
    func = lambda x: get_specified_alpha_stats(s, x['alpha_id'], x['simulate_data'], **simulation_config)
    with ThreadPool(3) as pool:
        for result in pool.map(
            func, result_list_flat
        ):
            stats_list_result.append(result)
            
    return stats_list_result 


def main():
    """
    Main function
    """

    s = start_session()

    k = [
        "vwap * 2",
        "open * close",
        "high * low",
        "vwap * 3",
        "open * close",
        "high * low",
    ]
    alpha_list = [generate_alpha(x) for x in k]

    simulate_alpha_list(s, alpha_list)


if __name__ == "__main__":
    main()
