# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# Copyright (c) 2024, RAYLEIGH RESEARCH OY. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Multi-Book High Frequency Trader Agent Class."""
import math
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.stats import rayleigh

from thesimulator import *
import re

#----------------------------------------------------------------------------
# Abstract class for high frequency trader agents.

class HighFrequencyTraderAgent:
    def __init__(self):
        self.recordedOrders = {}
        self.agentInventoryFile = None
        # (1) We track references to local data for each book.
        self.ask_lowest = dict()                                                                             
        self.bid_highest = dict() 
        self.priceHist0 = dict()
        self.priceHist = dict()
        self.ret = dict()
        self.deltaHFT = dict()
        self.tauHFT = dict()
        self.inventory = dict()
        self.base_free = dict()
        self.quote_free = dict()
        self.tauHFTFile = dict()
        self.agentId = 0 

    def log(self, message):
        print(f'T={self.currentTimestamp} : {self.name()} | {message}')

    def configure(self, simulation, params):
        # Initialize.
        print(f" --- Configuring {self.name()}--- ")
        self.currentTimestamp = simulation.currentTimestamp()
        self.logDir = simulation.logDir()
        self.bookCount = simulation.bookCount()
        self.duration = simulation.duration()
        self.xml = ET.parse(os.path.join(self.logDir, 'config.xml')).getroot()

        agent_config = self.xml.find("Agents").find("MultiBookExchangeAgent")
        self.priceDecimals = int(agent_config.attrib['priceDecimals'])
        self.volumeDecimals = int(agent_config.attrib['volumeDecimals'])
        self.maxLoan = int(agent_config.attrib['maxLoan'])
        self.maxLeverage = int(agent_config.attrib['maxLeverage'])

        self.debug = bool(int(params['debug']))
        self.exchange = str(params['exchange'])                                                                                                             # Exchange name
        # Cancellation horizon constant
        self.tau = float(params['tau'])                                                                                                                     # User-specified constant associated with time horizon of the high frequency trader agent
        # Max feed latency
        self.delta = float(params['delta'])
        # For optimal spread
        self.wealthFrac = 1
        # minimum feed latency
        self.minMFLatency = int(params['minMFLatency'])
        # Not used now, previously for cancellation
        # self.maxD  =int(params['maxD'])        
        # Order placement latency
        self.minOPLatency = int(params['minOPLatency'])
        self.maxOPLatency = int(params['maxOPLatency'])
        # Distribution shapes for latency
        self.alphaDelay = int(params['alphaDelay'])
        self.betaDelay = int(params['betaDelay'])
        # NOTE Remove above when Rayleigh latency is tested
        self.opLatencyScaleRay  = float(params['opLatencyScaleRay'])
        # # Parameters for order placement, notice that values should be log normalized
        # self.noiseSTD = float(params['noiseSTD'])
        # self.noiseMean = float(params['noiseMean'])
        # NOTE New thing for noise
        self.noiseRay = float(params['noiseRay'])
        self.orderMean = float(params['orderMean'])
        # Find the balance assigned to the HFT agents, 
        hftConfig = self.xml.find('Agents').find('HighFrequencyTraderAgent')
        self.baseInitial=float(hftConfig.find('Balances').find('Base').attrib['total'])                                                                                                       # User-specified constant associated with latency of the high frequency trader agent
        self.inventory = {i: 0 for i in range(self.bookCount)}
        # Inventory threshold
        self.psiHFT = {i : float(params['psiHFT_constant'])  for i in range(self.bookCount)} 
        # Risk aversion, gamma HFT, need to be scaled with variance of returns
        self.gHFT = float(params['gHFT'])                                                                                                                   # Constant determining the sensitivity to price and inventory changes
        sigmaScalingBase = int(agent_config.find('Books').find('Processes').attrib['updatePeriod'])
        # NOTE: For computational reasons better to leave squaring out
        self.sigmaSqrInit = float(agent_config.find('Books').find('Processes').find('GBM').attrib['sigma']) /(sigmaScalingBase/self.delta)
        # self.price0_seed = float(params['price0'])     
        # # Price of the asset at the beginning of simulation                                                                                                             # Tick size
        # # (2) Initialize historical data
        # for i in range(self.bookCount):
        #     gbm = GBM(X0=float(params['GBM_X0']), mu=float(params['GBM_mu']), sigma=float(params['GBM_sigma']),
        #                         lambda_jump=float(params['GBM_lambda_jump']), mu_jump=float(params['GBM_mu_jump']),
        #                         sigma_jump=float(params['GBM_sigma_jump']), flag_jump=bool(int(params['GBM_flag_jump'])),
        #                         seed=int(params['GBM_seed'])*(i+1))                                     
        #     _, Xt, _= gbm.price_series(T=1, N=int(self.duration / 1e9))
        #     self.priceHist0[i] = self.price0_seed *(1.+ Xt)
        #     self.priceHist[i] = self.priceHist0[i] 
        #     self.ret[i] = np.log(self.priceHist0[i][1:] / self.priceHist0[i][:-1])

        # (3) Prepare output files
        if self.debug:
            pattern = r'^HIGH_FREQUENCY_TRADER_AGENT_(?:[0-9]|1[0-9]|20)$'
            if bool(re.match(pattern, self.name())):
                self.agentInventoryFile = open(os.path.join(self.logDir,f"agent_inventory_{self.name()}.csv"), 'a')
                self.agentInventoryFile.write("Timestamp,BookId,Inventory,Free,Actual,qFree,qActual\n")
                self.agentInventoryFile.flush()

                self.orderHFT = open(os.path.join(self.logDir,f"orderHFT_{self.name()}.csv"), 'a')
                self.orderHFT.write("Timestamp,BookId,Price,Volume,orderID,ID,Direction,Leverage\n")
                self.orderHFT.flush()

                self.tradedOrder = open(os.path.join(self.logDir,f"tradedOrder_{self.name()}.csv"), 'a')
                self.tradedOrder.write("Timestamp,orderId,bookId,flag\n")
                self.tradedOrder.flush()

        # pattern = r'HIGH_FREQUENCY_TRADER_AGENT_(?:[0-9]|1[0-9]|20)$'
        # if bool(re.match(pattern, self.name())):
        #     self.tauHFTFile = open(os.path.join(self.logDir,f"tauHFT_{self.name()}.csv"), 'a')
        #     self.tauHFTFile.write(f"Timestamp,tau,bookId\n")
        #     self.tauHFTFile.flush()

    # receiveMessage method, providing communication with the simulator.
    def receiveMessage(self, simulation, messagetype, payload):
        self.currentTimestamp = simulation.currentTimestamp()
        # Subscribing to trade events occcurring within the simulator.
        #try:
        match messagetype:
                case "EVENT_SIMULATION_START":
                    self.log("-----SIMULATION STARTED----")
                    simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "SUBSCRIBE_EVENT_TRADE", EmptyPayload())                            
                    self.agentId = simulation.getAgentId(self.name())
                    for i in range(self.bookCount):
                        L1Payload = RetrieveL1Payload()
                        L1Payload.bookId = i   
                        simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)

                case "RESPONSE_SUBSCRIBE_EVENT_TRADE":
                    self.log("-----Subscribed to trade events----") 
                    for i in range(self.bookCount):
                        if self.debug:
                            self.log(f"BOOK {i} | BASE : {self.wealthFrac*float(simulation.account(self.name())[i].base.getFree())} | QUOTE : {self.wealthFrac*float(simulation.account(self.name())[i].quote.getFree())}")
                
                case "RESPONSE_RETRIEVE_L1":                    
                    bookId = payload.bookId
                    L1Payload = RetrieveL1Payload()
                    L1Payload.bookId = bookId
                    # Update the important values
                    self.deltaHFT[bookId] = self.delta / (1 + np.exp(abs(self.inventory[bookId]) - self.psiHFT[bookId])) 
                    # Not needed here
                    self.tauHFT[bookId] = max(self.tau*self.minMFLatency, int(np.ceil(self.tau * self.deltaHFT[bookId])))
                    # FIXME Later we need to check if there needs to be latency here as well if they are constantly active same time
                    simulation.dispatchMessage(self.currentTimestamp, int(max(self.deltaHFT[bookId],self.minMFLatency)), self.name(), self.exchange, "RETRIEVE_L1", L1Payload)
                    self.process_market_data(simulation, payload)

                case "RESPONSE_PLACE_ORDER_LIMIT":
                    np.random.seed()
                    if self.debug:
                        self.log(f"BOOK {payload.requestPayload.bookId} | {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} LIMIT ORDER #{payload.id} ACCEPTED : {payload.requestPayload.volume}@{payload.requestPayload.price}")
                    pattern = r'^HIGH_FREQUENCY_TRADER_AGENT_(?:[0-9]|1[0-9]|20)$'
                    if bool(re.match(pattern, self.name())):    
                            self.record_order(payload)
                    bookId = payload.requestPayload.bookId
                    cancel = CancelOrdersPayload([Cancellation(payload.id, None)],bookId)
                    self.deltaHFT[bookId] = self.delta / (1 + np.exp(abs(self.inventory[bookId]) - self.psiHFT[bookId])) 
                    # Cancellation time horizon
                    self.tauHFT[bookId] = max(self.tau*self.minMFLatency, int(np.ceil(self.tau * self.deltaHFT[bookId])))
                    simulation.dispatchMessage(self.currentTimestamp, int(self.tauHFT[bookId]), self.name(), self.exchange, "CANCEL_ORDERS", cancel)
                    #simulation.dispatchMessage(self.currentTimestamp, self.tauHFT[bookId], self.name(), self.exchange, "CANCEL_ORDERS", cancel)

                case "ERROR_RESPONSE_PLACE_ORDER_LIMIT":
                    bookId = payload.requestPayload.bookId
                    #print(f"{dir(payload.requestPayload)}|Error")
                    self.log(f"BOOK {bookId} | ERROR PLACING {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} LIMIT ORDER FOR {payload.requestPayload.volume}@{payload.requestPayload.price} : {payload.errorPayload.message}")
                    self.log(f"BOOK {bookId} | BID={self.bid_highest[bookId]} | ASK={self.ask_lowest[bookId]} | BASE : {self.wealthFrac*float(simulation.account(self.name())[bookId].base.getFree())} | QUOTE : {self.wealthFrac*float(simulation.account(self.name())[bookId].quote.getFree())}")

                case "RESPONSE_PLACE_ORDER_MARKET":
                    bookId = payload.requestPayload.bookId
                    if self.debug:
                        self.log(f"BOOK {bookId} | {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} MARKET ORDER #{payload.id} ACCEPTED FOR {payload.requestPayload.volume} ")

                case "ERROR_RESPONSE_PLACE_ORDER_MARKET":
                    bookId = payload.requestPayload.bookId
                    self.log(f"BOOK {bookId} | ERROR PLACING {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} MARKET ORDER FOR {payload.requestPayload.volume} : {payload.errorPayload.message}")
                    self.log(f"BOOK {bookId} | BID={self.bid_highest[bookId]} | ASK={self.ask_lowest[bookId]} | BASE : {self.wealthFrac*float(simulation.account(self.name())[bookId].base.getFree())} | QUOTE : {self.wealthFrac*float(simulation.account(self.name())[bookId].quote.getFree())}")

                case "RESPONSE_CANCEL_ORDERS":
                    bookId = payload.requestPayload.bookId
                    for cancel in payload.requestPayload.cancellations:
                        self.remove_order(bookId,int(cancel.id))
                    if self.debug:
                        self.log(f"BOOK {payload.requestPayload.bookId} | CANCELLED ORDERS : {','.join([str(c.id) for c in payload.requestPayload.cancellations])}")

                case "ERROR_RESPONSE_CANCEL_ORDERS":
                    bookId = payload.requestPayload.bookId            
                    if 'do not exist' not in payload.errorPayload.message:
                        self.log(f"BOOK {payload.requestPayload.bookId} |  ERROR CANCELLING ORDER : {payload.errorPayload.message}") 

                case "EVENT_TRADE":
                    bookId = int(payload.bookId)
                    if self.debug:
                        self.log(f"BOOK {payload.bookId} | BASE : {self.wealthFrac*float(simulation.account(self.name())[bookId].base.getFree())} | QUOTE : {self.wealthFrac*float(simulation.account(self.name())[bookId].quote.getFree())}")
                    # NOTE price history is not used currently
                    # spotPrice =  float(payload.trade.price()) 
                    # if spotPrice != self.priceHist[bookId][-1]:
                    #     self.priceHist[bookId] = np.append(self.priceHist[bookId], spotPrice)
                    #     self.priceHist[bookId] = self.priceHist[bookId]
                    #     self.ret[bookId] = np.append(self.ret[bookId],np.log(self.priceHist[bookId][-1]/self.priceHist[bookId][-2]))
                    #     #NOTE HARD CODE fix to not use too much memory
                    #     if len(self.ret[bookId]) > 500:
                    #         self.ret[bookId] = self.ret[bookId][-500:]
                    #         self.priceHist[bookId] = self.priceHist[bookId][-500:]
                    #     if self.debug:
                    #         self.log(' , '.join([str(p) for p in self.priceHist[bookId][-10:]]))
                    self.deltaHFT[bookId] = self.delta / (1 + np.exp(abs(self.inventory[bookId]) - self.psiHFT[bookId])) 
                    # self.tauHFT[bookId] = max(self.minMFLatency, int(np.ceil(self.tau * self.deltaHFT[bookId])))
                    self.process_trade_event(payload)

                case "EVENT_SIMULATION_STOP":
                    self.log("-----The simulation ends now----")

                case _:
                    self.log(messagetype)
        #except Exception as ex:
        #    self.log(f"MESSAGE FAILURE : {messagetype} : {ex}")

    
    def process_market_data(self, simulation, payload):
        bookId = payload.bookId
        self.ask_lowest[bookId] = float(payload.bestAskPrice)
        self.bid_highest[bookId] = float(payload.bestBidPrice)
        # (4) Avoid division by zero or missing data
        if (self.ask_lowest[bookId] <= 0) or (self.bid_highest[bookId] <= 0):
            return
        midPrice = (self.ask_lowest[bookId] + self.bid_highest[bookId]) / 2
        self.actual_spread = self.ask_lowest[bookId] - self.bid_highest[bookId]
        self.base_free[bookId] = self.wealthFrac*float(simulation.account(self.name())[bookId].base.getFree())
        self.quote_free[bookId] = self.wealthFrac*float(simulation.account(self.name())[bookId].quote.getFree())
        if self.debug:
            pattern = r'^HIGH_FREQUENCY_TRADER_AGENT_(?:[0-9]|1[0-9]|20)$'
            if bool(re.match(pattern, self.name())):
                self.agentInventoryFile.write(
                    f"{self.currentTimestamp},{bookId},{self.inventory[bookId]},{self.base_free[bookId]},0,{self.quote_free[bookId]},0\n"
                )
                self.agentInventoryFile.flush()

        self.deltaHFT[bookId] = self.delta / (1 + np.exp(abs(self.inventory[bookId]) - self.psiHFT[bookId]))    
        # self.tauHFT[bookId] = max(self.minMFLatency, int(np.ceil(self.tau * self.deltaHFT[bookId])))
        # pattern = r'^HIGH_FREQUENCY_TRADER_AGENT_(?:[0-9]|1[0-9]|20)$'
        # if bool(re.match(pattern, self.name())):
        #     self.tauHFTFile.write(f"{self.currentTimestamp},{self.tauHFT[bookId]},{bookId}\n")
        #     self.tauHFTFile.flush()


        self.priceReservation(midPrice, bookId)
        bidOrder, askOrder = self.placeorder_HFT(midPrice, bookId)
        ## Start canceling orders from the other side if we are at threshold and send only orders to other side
        if abs(self.inventory[bookId]) > self.psiHFT[bookId]: 
            if self.inventory[bookId] < 0:
                ## EXPERIMENTAL ##
                # m_volumeIncrement = 10**(-self.volumeDecimals)
                # vol = math.floor(bidOrder['Volume']/m_volumeIncrement)*m_volumeIncrement 
                # MarketOrderPayload = PlaceOrderMarketPayload(OrderDirection.Buy, vol, Decimal(0), int(bookId))
                # simulation.dispatchMessage(self.currentTimestamp, self.orderPlacementLatency(), self.name(), self.exchange, "PLACE_ORDER_MARKET", MarketOrderPayload)
                # self.cancelClosestToBestPrice(simulation, bookId, OrderDirection.Sell,float(payload.bestAskPrice))
                ## 

                self.send_order(simulation, bidOrder, bookId)
                # if np.random.uniform() < 0.5:
                self.cancelClosestToBestPrice(simulation, bookId, OrderDirection.Sell,float(payload.bestAskPrice))
                # else:
                #     self.send_order(simulation,askOrder,bookId, LimitOrderFlag.IOC)
            else:
                ## EXPERIMENTAL ##
                # vol = math.floor(askOrder['Volume']/m_volumeIncrement)*m_volumeIncrement 
                # MarketOrderPayload = PlaceOrderMarketPayload(OrderDirection.Buy, vol, Decimal(0), int(bookId))
                # simulation.dispatchMessage(self.currentTimestamp, self.orderPlacementLatency(), self.name(), self.exchange, "PLACE_ORDER_MARKET", MarketOrderPayload)
                # self.cancelClosestToBestPrice(simulation, bookId, OrderDirection.Buy,float(payload.bestBidPrice))
                self.send_order(simulation, askOrder, bookId)
                # if np.random.uniform() < 0.5:
                self.cancelClosestToBestPrice(simulation, bookId, OrderDirection.Buy,float(payload.bestBidPrice))
                # else:
                    # self.send_order(simulation,bidOrder,bookId,LimitOrderFlag.IOC)
        else:
            self.send_order(simulation, bidOrder, bookId)
            self.send_order(simulation, askOrder, bookId)

    def cancelClosestToBestPrice(self, simulation, bookId, direction, bestPrice, num_cancels = 1):
        """
        WIP: more than one cancel
        """
        closest = {'id': -1, 'delta': 999999}
        # There is ways to improve this later, now go through the recorded orders and check if they are correct side
        # and take which is closest to the bestPrice
        if bookId in self.recordedOrders:
            for book_orders in reversed(self.recordedOrders[bookId]):
                if book_orders['traded'] == 0 and book_orders['direction'] == direction: 
                    if closest['id'] == -1 or abs(book_orders['price'] - bestPrice) < closest['delta']:
                        closest = {'id': book_orders['order_id'], 'delta': abs(book_orders['price'] - bestPrice)}
            # If we found something, let us cancel as soon as possible
            if closest['id'] != -1:
                cancel = CancelOrdersPayload([Cancellation(closest['id'], None)],bookId)
                simulation.dispatchMessage(self.currentTimestamp, self.orderPlacementLatency(), self.name(), self.exchange, "CANCEL_ORDERS", cancel)
                                            
    # Method calculating the reservation price.
    def priceReservation(self, midQuotePrice, bookId):
        # Window should be adjusted to the new time scheme
        # window = self.tauHFT.get(bookId, 1)
        # print(f"THE WINDOW SIZE IS {window}")

        # if window < 10:
            # tau hft is wrong if we end up here
            # window = 10
        # revrets = self.ret[bookId][-window:] if len(self.ret[bookId]) >= window else self.ret[bookId]
        # if self.debug:
        #     self.log(f'{len(revrets)=}')
        #     self.log(f'{revrets=}')
        # self.var_returns = np.var(revrets)
        #self.pRes = midQuotePrice-self.gHFT*self.inventory[bookId]*var_returns*(self.duration-self.currentTimestamp) ##-----ALTERNATIVE FORMULA-----##
        #self.optimal_spread = self.gHFT*var_returns*(self.duration-self.currentTimestamp)+ (2/self.gHFT)*np.log(1 + (self.gHFT/self.kappa)) ##-----ALTERNATIVE FORMULA-----##
        sigmaSqr = self.sigmaSqrInit
        self.pRes = midQuotePrice-self.gHFT*self.inventory[bookId]*sigmaSqr # *(1-(self.currentTimestamp/self.duration))
        # self.optimal_spread = (2/self.gHFT)*np.log(1 + (self.gHFT/self.kappa))
        # print(f'{self.var_returns=}|{self.gHFT=}')

    def log_inputs(self, price_order, bookId):
        self.log(f'BOOK {bookId} | {self.pRes=}')
        self.log(f'BOOK {bookId} | {self.actual_spread=}')
        self.log(f'BOOK {bookId} | {bookId=}')
        self.log(f'BOOK {bookId} | {self.gHFT=}')
        self.log(f'BOOK {bookId} | {self.deltaHFT[bookId]=}')
        self.log(f'BOOK {bookId} | {self.tauHFT[bookId]=}')
        self.log(f'BOOK {bookId} | {self.bid_highest[bookId]=}')
        self.log(f'BOOK {bookId} | {self.ask_lowest[bookId]=}')
        self.log(f'BOOK {bookId} | {price_order=}')
        self.log(f'BOOK {bookId} | {self.gHFT*self.inventory[bookId]*self.var_returns=}')
        self.log(f'BOOK {bookId} | {self.gHFT*self.inventory[bookId]*self.var_returns*(1-(self.currentTimestamp/self.duration))=}')

    # Method placing orders for high frequency trading agents.
    def placeorder_HFT(self, midQuotePrice, bookId):
        np.random.seed()
        m_priceIncrement = 10**(-self.priceDecimals)
        # Shift for the Rayleigh
        # rayleighShift = self.noiseRay/4
        percentage = 0.03
        rayleighShift = self.noiseRay*np.sqrt(-2*np.log(1 - percentage))
        # Bid placement
        order_volume_bid = np.random.lognormal(mean=self.orderMean) 
        # Draw noise from Rayleigh distribution and shift by quarter mode
        noiseBid = rayleigh.rvs(scale=self.noiseRay) - rayleighShift 
        price_order_bid = self.pRes - (self.actual_spread / 2) - noiseBid
        # old noise component np.exp(np.random.normal(loc=self.noiseMean,scale=self.noiseSTD)) 
        limit_price_bid = np.round(price_order_bid / m_priceIncrement) * m_priceIncrement
        bidOrder = self.parse_HFT_order(bookId,limit_price_bid,order_volume_bid,'Buy')
        
        # Ask placement
        order_volume_ask = np.random.lognormal(mean=self.orderMean)
        noiseAsk = rayleigh.rvs(scale=self.noiseRay) - rayleighShift
        price_order_ask = self.pRes + (self.actual_spread / 2) + noiseAsk  
        # old noise component + np.exp(np.random.normal(loc=self.noiseMean,scale=self.noiseSTD))
        limit_price_ask = np.round(price_order_ask / m_priceIncrement) * m_priceIncrement
        askOrder = self.parse_HFT_order(bookId,limit_price_ask,order_volume_ask,'Sell')
       
        if self.debug:
            self.log_inputs(price_order_ask, bookId)
            self.log_inputs(price_order_bid, bookId)
        
        return bidOrder,askOrder

    def parse_HFT_order(self,bookId, limit_price, order_volume, direction):
        price = self.bid_highest[bookId] if direction == 'Sell' else self.ask_lowest[bookId]
        wealth = price*self.base_free[bookId] + self.quote_free[bookId]
        if limit_price <= 0 or order_volume <= 0 or wealth <= 0:
            return {'OrderType': 'NO_ORDER', 'Volume': 0, 'Leverage': 0, 'OrderDirection': direction}
        leverage = (order_volume*limit_price - wealth)/wealth
        if leverage > 0:
            if leverage > self.maxLeverage:
                    leverage = self.maxLeverage
            order_volume = order_volume/(1+leverage) 
            return {'OrderType': 'LIMIT_ORDER', 'Price': limit_price, 'Volume': order_volume, 'Leverage': leverage, 'OrderDirection': direction}
        else: 
            return {'OrderType': 'LIMIT_ORDER', 'Price': limit_price, 'Volume': order_volume, 'Leverage': 0, 'OrderDirection': direction}
                
        
    # Converting orders to data types readable by the simulator.
    def send_order(self, simulation, Order, bookId, flag=None):
        m_volumeIncrement = 10**(-self.volumeDecimals)
        if Order['OrderType'] == "NO_ORDER" or Order['Volume'] <= 0:
            return
        direction = OrderDirection.Buy if Order['OrderDirection'] == 'Buy' else OrderDirection.Sell
        adj_volume = math.floor(Order['Volume']/m_volumeIncrement)*m_volumeIncrement

        LimitOrderPayload = PlaceOrderLimitPayload(
                direction, 
                Decimal(Order['Volume']),
                Decimal(Order['Price'].item()),
                Decimal(Order['Leverage']),
                int(bookId)
        )
        # if flag:
            # pass
            # LimitOrderPayload.flag = flag
    
        simulation.dispatchMessage(
            self.currentTimestamp, self.orderPlacementLatency(),
            self.name(), self.exchange,
            "PLACE_ORDER_LIMIT", LimitOrderPayload
        )
        if self.debug:
            self.log(
                f"BOOK {bookId} | SUBMITTED {Order['OrderDirection']} ORDER "
                f"FOR {adj_volume}@{Order['Price']} WITH LEVERAGE {Order['Leverage']}"
            )

    def record_order(self, payload):
        bookId = payload.requestPayload.bookId
        self.recordedOrders.setdefault(bookId, []).append({
            'order_id': int(payload.id),
            'traded': 0, # Not needed anymore
            'price': float(payload.requestPayload.price),
            'volume': float(payload.requestPayload.volume),
            'direction': int(payload.requestPayload.direction)
        })
        if self.debug:
            self.orderHFT.write(
                f"{self.currentTimestamp},{bookId},{payload.requestPayload.price},"
                f"{payload.requestPayload.volume},{payload.id},"
                f"{payload.id},{payload.requestPayload.direction},{payload.requestPayload.leverage}\n"
            )
            self.orderHFT.flush()
    def remove_order(self, bookId, orderId, amount=None):
        if bookId not in self.recordedOrders:
            return
        for idx, order in enumerate(self.recordedOrders[bookId]):
            if order['order_id'] == orderId:
                if amount:
                    order['volume'] -= amount
                    if order['volume'] < 10**(-self.volumeDecimals):
                        self.recordedOrders[bookId].pop(idx)
                else:
                    self.recordedOrders[bookId].pop(idx)
                    return

    def process_trade_event(self, payload):
        bookId = payload.bookId
        if self.agentId == int(payload.context.aggressingAgentId):
            # NOTE Aggressive order is never recorded

            # Directions 0 buy initiated, 1 sell initated
            # HFT was aggressive => direction -1 (sell == 1) otherwise 1 
            direction = -1 if int(payload.trade.direction()) == 1 else 1
            self.inventory[bookId] += direction*float(payload.trade.volume())
        # FIXME elif when no self trading
        if self.agentId == int(payload.context.restingAgentId):
            # HFT was passive => direction 1 (other guy sold) otherwise -1
            direction = 1 if int(payload.trade.direction()) == 1 else -1
            self.inventory[bookId] += direction*float(payload.trade.volume())

            self.remove_order(bookId, int(payload.trade.restingOrderId()), float(payload.trade.volume()))
        # Get the aggressing & resting order IDs as floats for easy comparison
        # NOTE Removing trade tracking and only chance inventory
        # aggressing_id = int(payload.trade.aggressingOrderId())
        # resting_id = int(payload.trade.restingOrderId())
        # # Only proceed if we have any tracked orders in this book
        # if bookId not in self.recordedOrders:
        #     return
        # for order in self.recordedOrders[bookId]:
        #     if order["order_id"]==int(payload.trade.aggressingOrderId()) or self.agentId == int(payload.context.aggressingAgentId):
        #         # Directions 0 buy initiated, 1 sell initated
        #         # HFT was aggressive => direction -1 (sell == 1) otherwise 1 
        #         direction = -1 if int(payload.trade.direction()) == 1 else 1
        #         self.inventory[bookId] += direction*float(payload.trade.volume())
        #         order["traded"] = 1
        #         # Trade volume to CSV
        #         self.tradedOrder.write(f"{self.currentTimestamp},{aggressing_id},{bookId},{1},{round(float(payload.trade.volume()),4)}\n")
        #     # TODO put elif self trade is prevented
        #     if order["order_id"]==int(payload.trade.restingOrderId()):
        #         if self.agentId == 0:
        #             self.agentId = int(payload.context.aggressingAgentId)
        #         # HFT was passive => direction 1 (other guy sold) otherwise -1
        #         direction = 1 if int(payload.trade.direction()) == 1 else -1
        #         self.inventory[bookId] += direction*float(payload.trade.volume())
        #         # Trade volume to csv
        #         self.tradedOrder.write(f"{self.currentTimestamp},{resting_id},{bookId},{2},{round(float(payload.trade.volume()),4)}\n")

        # self.tradedOrder.flush() 
    
    def orderPlacementLatency(self):
        # beta_sample = np.random.beta(self.alphaDelay, self.betaDelay, size=1)
        ray_sample = rayleigh.rvs(scale=self.opLatencyScaleRay, size=1)
        latency = self.minOPLatency + (self.maxOPLatency-self.minOPLatency) * ray_sample
        latency = int(min(latency, self.maxOPLatency))
        return latency

    def rayleigh_latencies_ns(self, scale=10_000_000, seed=None):
        """
        WIP: replacee the above
        """
        batch = rayleigh.rvs(scale=scale, size=1)
        shifted = batch + self.minOPLatency
        shifted = shifted if shifted <= self.maxOPLatency else self.maxOPLatency
        return shifted
