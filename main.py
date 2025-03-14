import csv
import logging
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread

class WQSession(requests.Session):
    def __init__(self, json_fn='credentials.json'):
        super().__init__()
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)
        logging.basicConfig(encoding='utf-8', level=logging.INFO, format='%(asctime)s: %(message)s')
        self.json_fn = json_fn
        self.login()
        old_get, old_post = self.get, self.post
        def new_get(*args, **kwargs):
            try:
                return old_get(*args, **kwargs)
            except:
                return new_get(*args, **kwargs)
        def new_post(*args, **kwargs):
            try:
                return old_post(*args, **kwargs)
            except:
                return new_post(*args, **kwargs)
        self.get, self.post = new_get, new_post
        self.login_expired = False
        self.rows_processed = []

    def login(self):
        with open(self.json_fn, 'r') as f:
            creds = json.loads(f.read())
            email, password = creds['email'], creds['password']
            self.auth = (email, password)
            r = self.post('https://api.worldquantbrain.com/authentication')
        if 'user' not in r.json():
            if 'inquiry' in r.json():
                input(f"Please complete biometric authentication at {r.url}/persona?inquiry={r.json()['inquiry']} before continuing...")
                self.post(f"{r.url}/persona", json=r.json())
            else:
                print(f'WARNING! {r.json()}')
                input('Press enter to quit...')
        logging.info('Logged in to WQBrain!')

    def simulate(self, data):
        self.rows_processed = []

        def process_simulation(writer, f, simulation):
            if self.login_expired:
                return # expired crendentials
            thread = current_thread().name
            logging.info(f'{simulation["index"]}/{simulation["total"]} alpha simulations...')

            alpha = simulation['code'].strip()
            delay = simulation.get('delay', 1)
            universe = simulation.get('universe', 'TOP3000')
            truncation = simulation.get('truncation', 0.08)
            region = simulation.get('region', 'USA')
            decay = simulation.get('decay', 4)
            neutralization = simulation.get('neutralization', 'SUBINDUSTRY').upper()
            pasteurization = simulation.get('pasteurization', 'ON')
            nan = simulation.get('nanHandling', 'OFF')
            logging.info(f"{thread} -- Simulating alpha: {alpha}")
            while True:
                # keep sending a post request until the simulation link is found
                try:
                    r = self.post('https://api.worldquantbrain.com/simulations', json={
                        'regular': alpha,
                        'type': 'REGULAR',
                        'settings': {
                            "nanHandling": nan,
                            "instrumentType": "EQUITY",
                            "delay": delay,
                            "universe": universe,
                            "truncation": truncation,
                            "unitHandling": "VERIFY",
                            "pasteurization": pasteurization,
                            "region": region,
                            "language": "FASTEXPR",
                            "decay": decay,
                            "neutralization": neutralization,
                            "visualization": False
                        }
                    })
                    nxt = r.headers['Location']
                    break
                except:
                    logging.warning(f'{thread} -- Issue when sending simulation request:{r.content}')
                    try:
                        if 'credentials' in r.json()['detail']:
                            self.login_expired = True
                            return
                    except:
                        logging.info(f'{thread} -- {r.content}') # usually gateway timeout
                        return
            logging.info(f'{thread} -- Obtained simulation link: {nxt}')
            ok = True
            while True:
                r = self.get(nxt).json()
                if 'alpha' in r:
                    alpha_link = r['alpha']
                    break
                try:
                    logging.info(f"{thread} -- Waiting for simulation to end ({int(100*r['progress'])}%)")
                except Exception as e:
                    ok = (False, r['message'])
                    break
                time.sleep(10)
            if ok != True:
                logging.info(f'{thread} -- Issue when sending simulation request: {ok[1]}')
                row = [
                    0, delay, region,
                    neutralization, decay, truncation,
                    0, 0, 0, 'FAIL', 0, -1, universe, nxt, alpha
                ]
            else:
                r = self.get(f'https://api.worldquantbrain.com/alphas/{alpha_link}').json()
                logging.info(f'{thread} -- Obtained alpha link: https://platform.worldquantbrain.com/alpha/{alpha_link}')
                passed = 0
                for check in r['is']['checks']:
                    passed += check['result'] == 'PASS'
                    if check['name'] == 'CONCENTRATED_WEIGHT':
                        weight_check = check['result']
                    if check['name'] == 'LOW_SUB_UNIVERSE_SHARPE':
                        subsharpe = check['value']
                try:
                    subsharpe
                except:
                    subsharpe = -1
                # header = [
                #     'passed', 'delay', 'region', 'neutralization', 'decay', 'truncation',
                #     'sharpe', 'fitness', 'turnover', 'weight',
                #     'subsharpe', 'correlation', 'universe', 'link', 'code'
                # ]
                row = [
                    passed, delay, region, neutralization, decay, truncation,
                    r['is']['sharpe'], r['is']['fitness'], round(100*r['is']['turnover'], 2), weight_check,
                    subsharpe, -1, universe, f'https://platform.worldquantbrain.com/alpha/{alpha_link}', alpha
                ]
            writer.writerow(row)
            f.flush()
            self.rows_processed.append(simulation)
            logging.info(f'{thread} -- Result added to CSV!')

        try:
            for handler in logging.root.handlers:
                logging.root.removeHandler(handler)
            api_id = f"{str(time.time()).replace('.', '_')}"
            logging.basicConfig(encoding='utf-8', level=logging.INFO, format='%(asctime)s: %(message)s',
                                filename=f"data/api_{api_id}.log")
            logging.getLogger().addHandler(logging.StreamHandler())

            csv_file = f"data/api_{api_id}.csv"
            logging.info(f'Creating CSV file: {csv_file}')
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'passed', 'delay', 'region', 'neutralization', 'decay', 'truncation',
                    'sharpe', 'fitness', 'turnover', 'weight',
                    'subsharpe', 'correlation', 'universe', 'link', 'code'
                ]
                writer.writerow(header)
                with ThreadPoolExecutor(max_workers=3) as executor: # only 3 can go in concurrently
                    _ = executor.map(lambda sim: process_simulation(writer, f, sim), data)
        except Exception as e:
            print(f'Issue occurred! {type(e).__name__}: {e}')
        return [sim for sim in data if sim not in self.rows_processed]

if __name__ == '__main__':
    # 1. Construct DATA
    # from parameters import DATA

    import pandas as pd
    alpha_df = pd.read_excel('data/alphas.xlsx')
    alphas = alpha_df['func'].tolist()
    # print(alphas[:3])

    betas = [
        '-ts_delta(close, 2)',
        '-ts_delta(close, 5)',
        'ts_regression(ts_mean(volume, 2), ts_delta(close, 2), 90)',
        '-returns / volume',
        'trade_when(pcr_oi_270 < 1, (implied_volatility_call_270 - implied_volatility_put_270), -1)',
        '- scl12_buzz',
        'ts_std_dev(ts_backfill(snt_social_value, 60), 60)',
        'implied_volatility_call_120 / parkinson_volatility_120',
        'rank(ts_regression(fnd6_newqv1300_drcq, ts_step(1), 252, rettype=2))',
        '- ts_corr(ts_backfill(fscore_momentum, 66), ts_backfill(fscore_value, 66), 756)',
        'ts_rank(operating_income, 252)',
        'ts_backfill(vec_avg(nws12_prez_4l), 504)',
        '- ts_rank(fn_liab_fair_val_l1_a, 252)',
    ]

    DATA = []
    for i, a in enumerate(alphas):
        if i<6:
            continue
        for b in betas:
            code = f'zscore({b})+zscore({a})'
            # for univ in ['TOP1000', 'TOP500', 'TOP200', 'TOPSP500']:
            for univ in ['TOP3000']:
                # for n in ['NONE', 'MARKET', 'INDUSTRY', 'SUBINDUSTRY', 'SECTOR']:
                for n in ['SUBINDUSTRY']:
                    DATA.append({
                        'code': code,
                        'neutralization': n,
                        'region': 'USA',
                        'universe': univ,
                        'decay': 4,
                        'truncation': 0.08,
                        'delay': 1,
                    })

    # 2. Start Backfill
    TOTAL_ROWS = len(DATA)
    for i in range(TOTAL_ROWS):
        DATA[i]['index'] = i + 1
        DATA[i]['total'] = TOTAL_ROWS
    while DATA:
        wq = WQSession()
        # logging.info(f'{TOTAL_ROWS-len(DATA)}/{TOTAL_ROWS} alpha simulations...')
        DATA = wq.simulate(DATA)

