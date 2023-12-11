import sshtunnel
import pymongo
import datetime
import pandas as pd
import numpy as np
import websockets
import asyncio
import pdb
import matplotlib.pyplot as plt
from helpers import decompose_future_name
from plots import group_plot
import json
from bson.json_util import dumps # dump Output
#from src.vola_plots import trisurf, vola_surface_interpolated
#from src.helpers import decompose_instrument_name

# Todo: connection doesnt stop atm
class BRC:
    def __init__(self, collection_name):
        
        self.MONGO_HOST = '35.205.115.90'
        self.MONGO_DB   = 'cryptocurrency'
        self.MONGO_USER = 'winjules2'
        self.MONGO_PASS = ''
        self.PORT = 27017
        print('\nGoing Local')
        self.client = pymongo.MongoClient('localhost', 27017) 
        self.db = self.client[self.MONGO_DB]
        self.collection_name = collection_name #'deribit_transactions'
        self.collection = self.db[self.collection_name]
        print('using collection: ', self.collection_name)
        self._generate_stats()
        

    def _server(self):
        self.server = sshtunnel.SSHTunnelForwarder(
            self.MONGO_HOST,
            ssh_username=self.MONGO_USER,
            ssh_password=self.MONGO_PASS,
            remote_bind_address=('127.0.0.1', self.PORT)
            )
        return self.server

    def _start(self):
        #self.server = self._server()
        #self.server.start()
        return True

    def _stop(self):
        #self.server.stop()
        return True

    def _filter_by_timestamp(self, starttime, endtime):        
        """
        Example:
        starttime = datetime.datetime(2020, 4, 19, 0, 0, 0)
        endtime = datetime.datetime(2020, 4, 20, 0, 0, 0)
        """
        ts_high     = round(endtime.timestamp() * 1000)
        ts_low      = round(starttime.timestamp() * 1000)
        return ts_high, ts_low

    def _generate_stats(self):
        print('\n Established Server Connection')
        print('\n Size in GB: ', self.db.command('dbstats')['dataSize'] * 0.000000001) 
        
        # Get first and last element:
        last_ele = self.collection.find_one(
        sort=[( '_id', pymongo.DESCENDING )]
        )

        first_ele = self.collection.find_one(
            sort = [('_id', pymongo.ASCENDING)]
        )

        self.first_day = datetime.datetime.fromtimestamp(round(first_ele['timestamp']/1000))
        self.last_day  = datetime.datetime.fromtimestamp(round(last_ele['timestamp']/1000))
        self.first_day_timestamp = first_ele['timestamp']
        self.last_day_timestamp  = last_ele['timestamp']

        print('\n First day: ', self.first_day, ' \n Last day: ', self.last_day)

    def underlying(self, do_sample, write_to_file):
        """
        Extract high frequency prices of Deribit synthetic btc price
        """
        print('extracting synth index')
        if do_sample:
            pipeline = [
                {
                    "$sample": {"size": 120000},
                },

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d-%H-%M", "date": {'$toDate': '$timestamp' } }},
                            'avg_btc_price': {'$avg': '$underlying_price'}
                        }
                }

            ]
        else:
            pipeline = [
                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d-%H-%m", "date": {'$toDate': '$timestamp' } }},
                            'avg_btc_price': {'$avg': '$underlying_price'}
                        }
                }

            ]
        
        print('Pumping the Pipeline')
        synth_per_minute = self.collection.aggregate(pipeline)

        # Save Output as JSON
        a = list(synth_per_minute)
        j = dumps(a, indent = 2)
        
        if write_to_file:
            # Dump each element in a dict and save as JSON
            fname = "out/underlying_per_minute.JSON"
            print('Writing Output to ', fname)
            out = {}
            for ele in a:
                out[ele['_id']] = {'underlying': ele['avg_btc_price']}

            with open(fname,"w") as f:
                json.dump(out, f)
        else:
            out = json.loads(j)

        return out


    def _mean_iv(self, do_sample = False, write_to_file = False):
        """
        Task: 
            Select Average IV (for bid and ask) and group by day!

        Paste in the pipeline to have a sample for debugging
            {
                "$sample": {"size": 10},
            },
        """
        print('init mean iv')

        if do_sample:
            pipeline = [
                {
                    "$sample": {"size": 120000},
                },

                # Try to Subset / WHERE Statement
                #{'$match': {'bid_iv': {"$gt": 0.02}}},
                {'$match':
                {
                '$and': [
                    {'bid_iv': {'$gt': 0.01, '$lt': 200}},
                    {'ask_iv': {'$gt': 0.01, '$lt': 200}}
                ]
                }},

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d", "date": {'$toDate': '$timestamp' } }},
                            'avg_ask': {'$avg': '$ask_iv'},
                            'avg_bid': {'$avg': '$bid_iv'},
                            'avg_btc_price': {'$avg': '$underlying_price'}

                        }
                }

            ]


        
        else:
            pipeline = [
                # Try to Subset / WHERE Statement
                #{'$match': {'bid_iv': {"$gt": 0.02}}},
                {'$match':
                {
                '$and': [
                    {'bid_iv': {'$gt': 0.01, '$lt': 200}},
                    {'ask_iv': {'$gt': 0.01, '$lt': 200}}
                ]
                }},

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d", "date": {'$toDate': '$timestamp' } }},
                            'avg_ask': {'$avg': '$ask_iv'},
                            'avg_bid': {'$avg': '$bid_iv'},
                            'avg_btc_price': {'$avg': '$underlying_price'}

                        }
                }

            ]
        
        print('Pumping the Pipeline')
        avg_iv_per_day = self.collection.aggregate(pipeline)
        #print(list(avg_iv_per_day))

        # Save Output as JSON
        a = list(avg_iv_per_day)
        j = dumps(a, indent = 2)

        if write_to_file:
            # Dump each element in a dict and save as JSON
            fname = "/Users/julian/src/spd/out/volas_per_day.JSON"
            print('Writing Output to ', fname)
            out = {}
            for ele in a:
                out[ele['_id']] = {'ask': ele['avg_ask'],
                                    'bid': ele['avg_bid'],
                                    'underlying': ele['avg_btc_price']}

            with open(fname,"w") as f:
                json.dump(out, f)
        else:
            out = json.loads(j)

        return out

    def _summary_stats_preprocessed(self, dat, otm_thresh = 0.7, itm_thresh = 1.3, do_sample = True, write_to_file = True):
        """
        Task: 
            Summary Statistics for:
                Moneyness, Implied Volatility, Amount of Calls, Amount of Puts
                Per time-to-maturity in weeks (1,2,4,8)


        Paste in the pipeline to have a sample for debugging
            {
                "$sample": {"size": 10},
            },

        # Regex like Query
        db.users.find({'name': {'$regex': 'sometext'}})

        db.deribit_orderbooks.findOne({'instrument_name': {'$regex': '(?<=\-)(.*?)(?=\-)'}})
        """
        print('Pumping the Pipeline')

        """
        time_variable = 'timestamp' # timestamp

        collection_data = []
        documents = list(self.update_collection.find({time_variable:{'$exists': True}}).sort(time_variable).limit(1000))
        for doc in documents:
            collection_data.append(doc)

        while True:
            ids = set(doc['_id'] for doc in documents)
            cursor = self.update_collection.find({time_variable: {'$gte': documents[-1][time_variable]}})
            documents = list(cursor.limit(1000).sort(time_variable))
            if not documents:
                break  # All done.
            for doc in documents:
                # Avoid overlaps
                if doc['_id'] not in ids:
                    collection_data.append(doc)
        """

        #dat = #self.load_update_collection(do_sample = False)

        #dat = pd.DataFrame(collection_data)
        assert(dat.shape[0] != 0)
        df  = dat[['_id', 'strike', 'is_call', 'tau', 'mark_iv', 'date', 'moneyness']]    
        #df['date_short'] = df['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))

        # Clusters for Tau: 1, 2, 4, 8 Weeks
        df['nweeks'] = 0
        floatweek = 1/52
        df['nweeks'][(df['tau'] <= floatweek)] = 1
        df['nweeks'][(df['tau'] > floatweek) & (df['tau'] <= 2 * floatweek)] = 2
        df['nweeks'][(df['tau'] > 2 * floatweek) & (df['tau'] <= 3 * floatweek)] = 3
        df['nweeks'][(df['tau'] > 3 * floatweek) & (df['tau'] <= 4 * floatweek)] = 4
        df['nweeks'][(df['tau'] > 4 * floatweek) & (df['tau'] <= 8 * floatweek)] = 8

        # Table 1
        # Amount of OTM and ITM Options
        table1 = df[(df['moneyness'] < itm_thresh) & (df['moneyness'] > otm_thresh)].describe()

        # Moneyness and implied volatility of valid 50ETF options.
        table2 = df[['mark_iv', 'moneyness', 'nweeks']].groupby(['nweeks']).describe()

        # Table3 of Bitcoin gross returns
        #table3 = df[[]] # last per day
        # Just calculate this from the actual time series of index returns


        # options used for constructing implied volatility curves.
        table4 = df[['is_call', 'nweeks']].groupby(['nweeks']).describe()

        if write_to_file:
            table1.to_csv('summary_statistics_table1.csv')
            table2.to_csv('summary_statistics_table2.csv')
            table4.to_csv('summary_statistics_table4.csv')

        try:
            vola_surface_interpolated(df, out_path = 'out/volasurface/', moneyness_min = otm_thresh, moneyness_max = itm_thresh)
            vola_surface_interpolated(df, out_path = 'out/volasurface/restricted/', moneyness_min = 0.95, moneyness_max = 1.05)
        except Exception as e:
            print(e)
            #pdb.set_trace()

        return None

    
    def _summary_stats(self, otm_thresh, itm_thresh, do_sample = True, write_to_file = False):
        """
        Task: 
            Summary Statistics for:
                Moneyness, Implied Volatility, Amount of Calls, Amount of Puts
                Per time-to-maturity in weeks (1,2,4,8)


        Paste in the pipeline to have a sample for debugging
            {
                "$sample": {"size": 10},
            },

        # Regex like Query
        db.users.find({'name': {'$regex': 'sometext'}})

        db.deribit_orderbooks.findOne({'instrument_name': {'$regex': '(?<=\-)(.*?)(?=\-)'}})
        """
        print('init mean iv')
        # db.deribit_orderbooks.regexFind({'ext':{'input':'$instrument_name', 'regex':'(?<=\-)(.*?)(?=\-)'})}

        
        print('Pumping the Pipeline')
        #cursor = self.collection.find(no_cursor_timeout=True)
        # Save Output as JSON
        #collection_data = [document for document in cursor]

        collection_data = []
        documents = list(self.collection.find().sort('timestamp').limit(1000))
        for doc in documents:
            collection_data.append(doc)

        while True:
            ids = set(doc['_id'] for doc in documents)
            cursor = self.collection.find({'timestamp': {'$gte': documents[-1]['timestamp']}})
            documents = list(cursor.limit(1000).sort('timestamp'))
            if not documents:
                break  # All done.
            for doc in documents:
                # Avoid overlaps
                if doc['_id'] not in ids:
                    collection_data.append(doc)


        dat = pd.DataFrame(collection_data)

        assert(dat.shape[0] != 0)

        # Convert dates, utc
        dat['date'] = list(map(lambda x: datetime.datetime.fromtimestamp(x/1000), dat['timestamp']))
        dat_params  = decompose_instrument_name(dat['instrument_name'], dat['date'])
        dat         = dat.join(dat_params)

        # Drop all spoofed observations - where timediff between two orderbooks (for one instrument) is too small
        dat['timestampdiff'] = dat['timestamp'].diff(1)
        dat = dat[(dat['timestampdiff'] > 2)]

        dat['interest_rate'] = 0 # assumption here!
        dat['index_price']   = dat['index_price'].astype(float)

        # To check Results after trading 
        dates                       = dat['date']
        dat['strdates']             = dates.dt.strftime('%Y-%m-%d') 
        maturitydates               = dat['maturitydate_trading']
        dat['maturitydate_char']    = maturitydates.dt.strftime('%Y-%m-%d')

        # Calculate mean instrument price
        bid_instrument_price = dat['best_bid_price'] * dat['underlying_price'] 
        ask_instrument_price = dat['best_ask_price'] * dat['underlying_price']
        dat['instrument_price'] = (bid_instrument_price + ask_instrument_price) / 2

        # Prepare for moneyness domain restriction (0.8 < m < 1.2)
        dat['moneyness']    = round(dat['strike'] / dat['index_price'], 2)
        df                  = dat[['_id', 'index_price', 'strike', 'interest_rate', 'maturity', 'is_call', 'tau', 'mark_iv', 'date', 'moneyness', 'instrument_name', 'days_to_maturity', 'maturitydate_char', 'timestamp', 'underlying_price', 'instrument_price']]    
        
        # Select Tau and Maturity (Tau is rounded, prevent mix up!)
        unique_taus = df['tau'].unique()
        unique_maturities = df['maturity'].unique()
        
        # Save Tau-Maturitydate combination
        #tau_maturitydate[curr_day.strftime('%Y-%m-%d')] = (unique_taus,)
        
        unique_taus.sort()
        unique_taus = unique_taus[(unique_taus > 0) & (unique_taus < 0.25)]
        print('\nunique taus: ', unique_taus,
                '\nunique maturities: ', unique_maturities)

        # Clusters for Tau: 1, 2, 4, 8 Weeks
        df['nweeks'] = 0
        floatweek = 1/52
        df['nweeks'][(df['tau'] <= floatweek)] = 1
        df['nweeks'][(df['tau'] > floatweek) & (df['tau'] <= 2 * floatweek)] = 2
        df['nweeks'][(df['tau'] > 2 * floatweek) & (df['tau'] <= 3 * floatweek)] = 3
        df['nweeks'][(df['tau'] > 3 * floatweek) & (df['tau'] <= 4 * floatweek)] = 4
        df['nweeks'][(df['tau'] > 4 * floatweek) & (df['tau'] <= 8 * floatweek)] = 8

        # Table 1
        # Amount of OTM and ITM Options
        table1 = df[(df['moneyness'] < itm_thresh) & (df['moneyness'] > otm_thresh)]

        # Moneyness and implied volatility of valid 50ETF options.
        table2 = df[['mark_iv', 'moneyness', 'nweeks']].groupby(['nweeks']).describe()

        # Table3 of Bitcoin gross returns is calculated in the .Rmd

        # options used for constructing implied volatility curves.
        table4 = df[['is_call', 'nweeks']].groupby(['nweeks']).describe()

        if write_to_file:
            table1.to_csv('summary_statistics_table1.csv')
            table2.to_csv('summary_statistics_table2.csv')
            table4.to_csv('summary_statistics_table4.csv')

        return None

    def _run(self, starttime, endtime):
        
        #server_started = self._start()
        
    
        try:

            download_starttime = datetime.datetime.now()

            ts_high, ts_low = self._filter_by_timestamp(starttime, endtime)

            _filter = { "$and": [{'timestamp': {"$lt": ts_high}},
                                {'timestamp': {"$gte": ts_low}}]}
            res = self.collection.find(_filter)#.sort('timestamp')

            out = []
            for doc in res:
                out.append(doc)

            return out

        except Exception as e:
            print('Error: ', e)
            print('\nDisconnecting Server within error handler')
            self._stop()
            self.client.close()

        # Deribit
    def create_msg(self, _tshigh, _tslow, instrument_name):
        # retrieves constant interest rate for time frame
        self.msg = \
        {
        "jsonrpc" : "2.0",
        "id" : None,
        "method" : "public/get_funding_rate_value",
        "params" : {
            "instrument_name" : instrument_name, #"BTC-PERPETUAL"
            "start_timestamp" : _tslow,
            "end_timestamp" : _tshigh
            }
        }
        return None

    def create_msg2(self,instrument_name):
        msg = \
        {"jsonrpc": "2.0",
        "method": "public/get_funding_chart_data",
        "id": None,
        "params": {
            "instrument_name": instrument_name,
            "length": "8h"}
        }
        return None

    async def call_api(self, test = False):
        if test:
            url = 'wss://test.deribit.com/ws/api/v2'
            print('Using TEST API!')
        else:
            url = 'wss://www.deribit.com/ws/api/v2'
            print('Using LIVE API!')
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps(self.msg))
            while websocket.open:
                print(self.msg)
                response = await websocket.recv()
                # do something with the response...
                self.response = json.loads(response)
                self.historical_interest_rate = round(self.response['result'], 8)
                return None


    def download_historical_funding_rate(self, starttime, endtime, instrument_name):
        """

        """
        ts_high, ts_low = self._filter_by_timestamp(starttime, endtime)
        # This is the 8hour funding rate
        try:
            self.create_msg(ts_high, ts_low, instrument_name)
            asyncio.get_event_loop().run_until_complete(self.call_api())
        except Exception as e:
            print('Error while downloading from Deribit: ', e)
            print('Proceeding with interest rate of None')
            self.historical_interest_rate = None
        finally: 
            return self.historical_interest_rate

    def download_other_funding_rate(self, instrument_name):
        """

        """
        # This is the 8hour funding rate
        try:
            self.create_msg2(instrument_name)
            asyncio.get_event_loop().run_until_complete(self.call_api())
        except Exception as e:
            print('Error while downloading from Deribit: ', e)
            print('Proceeding with interest rate of None')
            self.historical_interest_rate = None
        finally: 
            pdb.set_trace()
            return self.historical_interest_rate




if __name__ == '__main__':
    #brc = BRC(collection_name='deribit_transactions')
    #fund = brc.download_other_funding_rate(datetime.datetime(2023, 11, 25, 0, 0, 0), datetime.datetime(2023, 11, 25, 18, 0, 0), 'BTC-PERPETUAL')
    brc = BRC(collection_name='deribit_futures_transactions')
    out = brc._run(datetime.datetime(2017, 1, 20, 0, 0, 0), datetime.datetime(2023, 11, 25, 0, 0, 0))
    df = pd.DataFrame(out).drop_duplicates()
    df['date'] = list(map(lambda x: datetime.datetime.fromtimestamp(x/1000), df['timestamp']))
    dat_params = decompose_future_name(df['instrument_name'], df['date'])
    df         = df.join(dat_params).sort_values('timestamp')
    df['premium'] = df['price'] - df['index_price']
    df['pct_premium'] = df['premium'] / df['index_price']
    df['annualized_premium'] = df.apply(lambda x: ((1 + x['pct_premium']) ** (x['tau']**(-1))) - 1 if x['tau'] >= 0.01 else None, axis = 1)
    pdb.set_trace()
    
    group_plot(df, 'date', 'price', 'instrument_name', 'out/futures_prices.png')
    group_plot(df, 'date', 'mark_price', 'instrument_name', 'out/futures_mark_prices.png')
    group_plot(df, 'date', 'premium', 'instrument_name', 'out/futures_premia.png')
    group_plot(df, 'date', 'pct_premium', 'instrument_name', 'out/futures_pct_premia.png')
    group_plot(df, 'date', 'annualized_premium', 'instrument_name', 'out/futures_annualized_premia.png')

    pdb.set_trace()
    

    fund = brc.download_historical_funding_rate(datetime.datetime(2023, 11, 25, 0, 0, 0), datetime.datetime(2023, 11, 25, 21, 0, 0), 'ETH-PERPETUAL')
    print(fund)
    pdb.set_trace()
    brc.download_other_funding_rate('BTC-PERPETUAL')
    #dat, interest_rate = brc._run(datetime.datetime(2022, 1, 1, 0, 0, 0),
    #               datetime.datetime(2022, 4, 1, 0, 0, 0),
    #                False, False, '')
    #print(len(dat))
    brc._summary_stats_preprocessed()
    #brc.plot_iv_surface()
    #d = brc._summary_stats(1.1, 0.9)

    #d = pd.DataFrame(dat)
    #d.to_csv('data/orderbooks_test.csv')
    #print('here')   