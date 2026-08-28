# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# Copyright (c) 2024, RAYLEIGH RESEARCH OY. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Multi-Book Stylized Trader Agent Class."""
import os
import re
import math
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from GBM import GBM
import xml.etree.ElementTree as ET
from scipy.stats import nbinom, rayleigh

from thesimulator import *

#----------------------------------------------------------------------------
# Abstract class for multi-book stylized trader agents.

class StylizedTraderAgent:
    def log(self, message):
        print(f'T={self.currentTimestamp} : {self.name()} | {message}')


    def configure(self, simulation, params):
        """
        Configures the agent with simulation parameters.
        """
        print(f" --- Configuring {self.name()}--- ")
        self.currentTimestamp = simulation.currentTimestamp()
        self.logDir = simulation.logDir()
        self.duration = simulation.duration()
        self.bookCount = simulation.bookCount()

        # Load XML configuration.
        self.xml = ET.parse(os.path.join(self.logDir, 'config.xml')).getroot()
        self.priceDecimals = int(self.xml.find("Agents").find("MultiBookExchangeAgent").attrib['priceDecimals'])
        self.volumeDecimals = int(self.xml.find("Agents").find("MultiBookExchangeAgent").attrib['volumeDecimals'])
        self.baseDecimals = int(self.xml.find("Agents").find("MultiBookExchangeAgent").attrib['baseDecimals'])
        self.quoteDecimals = int(self.xml.find("Agents").find("MultiBookExchangeAgent").attrib['quoteDecimals'])

        # Agent parameters
        self.debug = bool(int(params['debug']))                                                  # Debug mode
        self.regimeChangeFlag = bool(int(params['regimeChangeFlag']))
        self.exchange = str(params['exchange'])                                                  # Exchange name
        self.sigmaF = float(params['sigmaF'])                                                    # Standard deviation for fundamentalist weight component
        self.sigmaC = float(params['sigmaC'])                                                    # Standard deviation for chartist weight component
        self.sigmaN = float(params['sigmaN'])                                                    # Standard deviation for noise (zero-intelligence) weight component
        self.sigmaFRegime = float(params['sigmaFRegime'])                                        # ALTERNATIVE REGIME: Standard deviation for fundamentalist weight component 
        self.sigmaCRegime = float(params['sigmaCRegime'])                                        # ALTERNATIVE REGIME: Standard deviation for chartist weight component 
        self.sigmaNRegime = float(params['sigmaNRegime'])                                        # ALTERNATIVE REGIME: Standard deviation for noise (zero-intelligence) weight component                                                                       
        self.price0_seed = float(params['price0'])                                               # Price of the asset at the beginning of simulation
        self.timeScale =  self.timeScaleConverter(self.xml.get('timescale'))                     # Reference time unit 
        self.tau = float(params['tau'])                                       # User-specified constant associated with time horizon of the stylized trader agent
        self.tauF = float(params['tauF'])                                                        # Time constant associated with the fundamentalist components
        self.sigmaEps = float(params['sigmaEps'])                                                # Standard deviation of the noise induced component
        self.r_aversion = float(params['r_aversion'])                                            # User-specified constant associated with the risk aversion of the stylized trader agent
        self.placeOrderDelay = 1

        # Wealth and trade initializations.
        self.Stock_t = dict()
        self.Cash_t = dict()
        self.ask_lowest = dict()
        self.bid_highest = dict()
        self.tradeEvent = dict()

        np.random.seed()
        self.regimeProb = float(params['regimeProb'])
        self.wealthFrac = 1
        # self.minD = int(params['minD'])
        # self.maxD  =int(params['maxD'])
        self.alphaDelay = int(params['alphaDelay'])
        self.betaDelay = int(params['betaDelay'])
        # NOTE Remove above when Rayleigh latency is tested
        self.opLatencyScaleRay  = float(params['opLatencyScaleRay'])

        self.minOPLatency = int(params['minOPLatency'])
        self.maxOPLatency = int(params['maxOPLatency'])
        # Scale for Rayleigh distribution how many active agents at same time
        self.scaleR = float(params['scaleR'])
        # Market feed latency components
        self.mflMean = int(params['MFLmean'])
        self.mflSTD = int(params['MFLstd'])
        # Techincal latency component
        self.delayMean = int(params['delayMean'])
        self.delaySTD = int(params['delaySTD'])
        # Set initial weights using Laplace distribution.
        self.fundWeight = abs(np.random.laplace(loc = self.sigmaF, scale = self.sigmaF))         # Weight associated with the fundamentalist component
        self.chartWeight = abs(np.random.laplace(loc = self.sigmaC, scale = self.sigmaC))        # Weight associated with the chartist component
        self.noiseWeight = abs(np.random.laplace(loc = self.sigmaN, scale = self.sigmaN))        # Weight associated with the noise induced component
        self.fundWeightOriginal = self.fundWeight
        self.chartWeightOriginal = self.chartWeight
        self.noiseWeightOriginal = self.noiseWeight
        
        self.fcastAdjust = 1. / (self.fundWeight + self.chartWeight + self.noiseWeight)          # Normalizing constant
        self.tau_agent = int(np.ceil(self.tau * (1 + self.fundWeight) / (1 + self.chartWeight))) # Investment time horizon of the stylized trader agent
        self.r_aversion_agent = self.r_aversion * (1 + self.fundWeight)/(1 + self.chartWeight)   # Risk aversion of the stylized trader agent
        self.price0 = {bookId : self.price0_seed for bookId in range(self.bookCount)}            # Price of the assset at the beginning of historical data
        self.price = self.price0.copy()                                                          # Price of the assset at timestamp t
        self.orderFlag = {i : False for i in range(self.bookCount)}      
        self.Tinit= self.duration
        
        # NOTE: horizon extended for longer tests.
        self.n_trades = max(20, int(np.ceil(50* (1 + self.fundWeight) / (1 + self.chartWeight))))

        # Historical price and return initializations
        self.priceHist0 = dict()
        self.priceHist = dict()
        self.ret = dict()

        # Generate price histories using GBM.
        GBM_price = dict()
        Xt = dict()
        for i in range(self.bookCount):
            GBM_price[i] = GBM(
                X0=float(params['GBM_X0']),
                mu=float(params['GBM_mu']),
                sigma=float(params['GBM_sigma']),
                lambda_jump=float(params['GBM_lambda_jump']),
                mu_jump=float(params['GBM_mu_jump']),
                sigma_jump=float(params['GBM_sigma_jump']),
                flag_jump=bool(int(params['GBM_flag_jump'])),
                seed=int(params['GBM_seed'])*(i+1)
            )                                     
            #_, Xt[i], _= GBM_price[i].price_series(T=1, N=self.Tinit)
            _, Xt[i], _= GBM_price[i].price_series(T = 1, N = int(self.Tinit * self.timeScale))
            self.priceHist0[i] = self.price0[i] *(1. + Xt[i])
            self.priceHist[i] = self.priceHist0[i].copy()
            # Compute returns from the price history                                                
            self.ret[i] = Xt[i][0]                                                               # Price of the asset over the last tau timestamps at timestamp t
            self.ret[i]=np.append(
                self.ret[i], 
                np.log(self.priceHist0[i][1:]/self.priceHist0[i][:-1])
            )                                                                                    # Log-Return of the asset over the last Tiniti timestamps at the beginning of simulations

        # Parameters for NB-ACD models.
        # self.omegaACD = {i : float(params['omegaACD']) for i in range(self.bookCount)}
        # self.alphaACD = {i : float(params['alphaACD']) for i in range(self.bookCount)}
        # self.betaACD = {i : float(params['betaACD']) for i in range(self.bookCount)}
        # self.rACD = {i : float(params['rACD']) for i in range(self.bookCount)}
        # self.psiACD = self.omegaACD.copy()

        # Read agent count information from XML
        self.cCount = 0
        self.fCount = 0
        self.hCount = 0
        self.nCount = 0
        for child in self.xml.find("Agents"):
            if child.tag == 'InitializationAgent':
                self.iCount = int(child.attrib['instanceCount'])
            if child.tag == 'StylizedTraderAgent':
                if (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaC'])!= 0) or \
                   (float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0) or \
                   (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0) or \
                   (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0): 
                    self.sCount = int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaF']) != 0:
                    self.fCount = int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaC']) != 0:
                    self.cCount = int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaN']) != 0:
                    self.nCount = int(child.attrib['instanceCount'])
            if child.tag == 'HighFrequencyTraderAgent':
                self.hCount = int(child.attrib['instanceCount'])

        self.activityH = 1
        self.activityS = 1
        self.activityF = 1
        self.activityC = 1
        self.activityN = 1
 
    # ------------------------- Message Handling ------------------------- #
    def receiveMessage(self, simulation, messagetype, payload):
        """
        Dispatches messages to the corresponding handler functions.
        """
        self.currentTimestamp = simulation.currentTimestamp()
        # Update regime weights if specified by self.regimeChangeFlag
        if self.regimeChangeFlag:
            np.random.seed()
            rnd_num = np.random.uniform()
            if rnd_num <= self.regimeProb:
                self.fundWeight = np.abs(np.random.laplace(loc = self.sigmaFRegime, scale = self.sigmaFRegime))  
                self.chartWeight= np.abs(np.random.laplace(loc = self.sigmaCRegime, scale = self.sigmaCRegime)) 
                self.noiseWeight= np.abs(np.random.laplace(loc = self.sigmaNRegime, scale = self.sigmaNRegime)) 
                self.fcastAdjust = 1. / (self.fundWeight + self.chartWeight + self.noiseWeight) 
                self.tau_agent = int(np.ceil(self.tau * (1 + self.fundWeight) / (1 + self.chartWeight)))
 
                self.r_aversion_agent = self.r_aversion * (1 + self.fundWeight)/(1 + self.chartWeight)
            else:
                self.fundWeight = self.fundWeightOriginal
                self.chartWeight = self.chartWeightOriginal
                self.noiseWeight = self.noiseWeightOriginal

        # Subscribing to trade events occcurring within the simulator.
        #try:
        match messagetype:
                case "EVENT_SIMULATION_START":
                    self.handle_simulation_start(simulation, payload)
                case "RESPONSE_SUBSCRIBE_EVENT_TRADE":
                    self.handle_subscribe_event_trade(simulation, payload)
                case "RESPONSE_RETRIEVE_L1":
                    self.handle_response_retrieve_l1(simulation, payload) 
                case "RESPONSE_PLACE_ORDER_LIMIT":
                    self.handle_response_place_order_limit(simulation, payload)
                case "ERROR_RESPONSE_PLACE_ORDER_LIMIT":
                    self.handle_error_response_place_order_limit(simulation, payload)
                case "RESPONSE_PLACE_ORDER_MARKET":
                    self.handle_response_place_order_market(simulation, payload)
                case "ERROR_RESPONSE_PLACE_ORDER_MARKET":
                    self.handle_error_response_place_order_market(simulation, payload)
                case "RESPONSE_CANCEL_ORDERS":
                    self.handle_response_cancel_orders(simulation, payload) 
                case "ERROR_RESPONSE_CANCEL_ORDERS":
                    self.handle_error_response_cancel_orders(simulation, payload)
                case "EVENT_TRADE":
                    self.handle_event_trade(simulation, payload)    
                case "EVENT_SIMULATION_STOP":
                    self.handle_event_simulation_stop(simulation, payload)
                case _:
                    self.log(messagetype)
        #except Exception as ex:
        #    self.log(f"MESSAGE FAILURE : {messagetype} : {ex}")

    # ------------------------- Message Handling Methods ------------------------- #
    def handle_simulation_start(self, simulation, payload):
        self.log("-----SIMULATION STARTED----")
        simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "SUBSCRIBE_EVENT_TRADE", EmptyPayload())
        for i in range(self.bookCount):
            L1Payload = RetrieveL1Payload()
            L1Payload.bookId = i   
            simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)
    
    def handle_subscribe_event_trade(self, simulation, payload):
        self.log("-----Subscribed to trade events----")  
        for i in range(self.bookCount):
            L1Payload = RetrieveL1Payload()
            L1Payload.bookId = i    
            simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)
            if self.debug:
                account = simulation.account(self.name())[i]
                self.log(f"BOOK {i} | BASE : {self.wealthFrac*float(account.base.getFree())} | QUOTE : {self.wealthFrac*float(account.quote.getFree())}")

    def handle_response_retrieve_l1(self, simulation, payload):
        bookId = payload.bookId
        print(f'BOOK: {bookId} | AGENT: {self.name()} | RECEIVED A RESPONSE_RETRIEVE_L1 MESSAGE FROM THE EXCHAGE AT TIMESTAMP {self.currentTimestamp}')

        L1Payload = RetrieveL1Payload()
        L1Payload.bookId = bookId
        # Debug if needed print(f'[**] feedLatency {feedLatency}')
        # StylishedTraders are delayed by marketFeedLatency and Technical Delay in the decision
        dispatchMessageDelay =  self.marketFeedLatency() + self.decisionMakingDelay()
        simulation.dispatchMessage(self.currentTimestamp,dispatchMessageDelay, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)

        self.ask_lowest[bookId] = float(payload.bestAskPrice)
        self.bid_highest[bookId] = float(payload.bestBidPrice)
        midPrice = (self.ask_lowest[bookId] + self.bid_highest[bookId]) / 2
        # NOTE very unlikely that timestamp has trade so let's make estimation that last trade is with in a second
        if bookId in self.tradeEvent and abs(self.tradeEvent[bookId][1] - self.currentTimestamp) < 1e8:
            spotPrice = self.tradeEvent[bookId][0]
        else:
            spotPrice = midPrice

        # Update price history
        # if (self.ask_lowest[bookId] != 0 and self.bid_highest[bookId] != 0):
            # self.priceHist[bookId] = np.append(self.priceHist[bookId], self.tradeEvent[(self.currentTimestamp,bookId)])
            # self.priceHist[bookId] = self.priceHist[bookId][-int(self.Tinit * self.timeScale)-1:]
            # self.ret[bookId] = np.append(self.ret[bookId][1:],np.log(self.priceHist[bookId][-1]/self.priceHist[bookId][-2]))
            # if len(self.ret[bookId]) > self.tau_agent:
                # self.ret[bookId] = self.ret[bookId][-self.tau_agent:]
            # if spotPrice != self.priceHist[bookId][-1]:
            #     self.priceHist[bookId] = np.append(self.priceHist[bookId], spotPrice)
            #     #self.priceHist[bookId] = self.priceHist[bookId][-self.Tinit-1:]
            #     self.priceHist[bookId] = self.priceHist[bookId][-int(self.Tinit * self.timeScale)-1:]
            #     self.ret[bookId] = np.append(self.ret[bookId][1:],np.log(self.priceHist[bookId][-1]/self.priceHist[bookId][-2]))
            #     if self.debug:
            #         self.log(' , '.join([str(p) for p in self.priceHist[bookId][-10:]])) 

        if self.selectNextAgent(bookId, int(self.currentTimestamp//1e4)):   
            print(f'SELECTED AGENT IS {self.name()} | CURRENT TIMESTAMP IS {self.currentTimestamp}') 
            if not self.orderFlag[bookId] and (self.ask_lowest[bookId] != 0 and self.bid_highest[bookId] != 0): 
                account = simulation.account(self.name())[bookId]   
                self.Stock_t[bookId] = self.wealthFrac*float(account.base.getFree())
                self.Cash_t[bookId] = self.wealthFrac*float(account.quote.getFree())
                self.priceFcast(price = spotPrice, priceF = simulation.processValue("fundamental", bookId), bookId = bookId)
                self.price = spotPrice
                Order = self.placeorder_STA(simulation, ask_lowest = self.ask_lowest[bookId] , bid_highest = self.bid_highest[bookId], Timestamp = self.currentTimestamp, bookId=bookId)
                orderLatency = self.orderPlacementLatency()
                self.orderconvertsimulator_STA(simulation, Order, bookId=bookId, placeOrderLatency=orderLatency)
     
    def handle_response_place_order_limit(self, simulation, payload):
        self.log(f"BOOK {payload.requestPayload.bookId} | {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} LIMIT ORDER #{payload.id} ACCEPTED : {payload.requestPayload.volume}@{payload.requestPayload.price}")
        bookId = payload.requestPayload.bookId
        cancel = CancelOrdersPayload([Cancellation(payload.id, None)], bookId)
        cancel.bookId = bookId 
        simulation.dispatchMessage(self.currentTimestamp, self.tau_agent, self.name(), self.exchange, "CANCEL_ORDERS", cancel)
        self.orderFlag[bookId] = False
    
    def handle_error_response_place_order_limit(self, simulation, payload):
        bookId = payload.requestPayload.bookId
        self.log(f"BOOK {bookId} | ERROR PLACING {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} LIMIT ORDER FOR {payload.requestPayload.volume}@{payload.requestPayload.price} : {payload.errorPayload.message}")
        self.log(f"BOOK {bookId} | BID={self.bid_highest[bookId]} | ASK={self.ask_lowest[bookId]} | BASE : {self.wealthFrac*float(simulation.account(self.name())[bookId].base.getFree())} | QUOTE : {self.wealthFrac*float(simulation.account(self.name())[bookId].quote.getFree())}")
        self.orderFlag[bookId] = False

    def handle_response_place_order_market(self, simulation, payload):
        bookId = payload.requestPayload.bookId
        self.log(f"BOOK {bookId} | {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} MARKET ORDER #{payload.id} ACCEPTED FOR {payload.requestPayload.volume} ")
        self.orderFlag[bookId] = False

    def handle_error_response_place_order_market(self, simulation, payload):
        bookId = payload.requestPayload.bookId
        self.log(f"BOOK {bookId} | ERROR PLACING {'BUY' if payload.requestPayload.direction == OrderDirection.Buy else 'SELL'} MARKET ORDER FOR {payload.requestPayload.volume} : {payload.errorPayload.message}")
        account = simulation.account(self.name())[bookId]
        self.log(f"BOOK {bookId} | BID={self.bid_highest[bookId]} | ASK={self.ask_lowest[bookId]} | BASE : {self.wealthFrac*float(account.base.getFree())} | QUOTE : {self.wealthFrac*float(account.quote.getFree())}")
        self.orderFlag[bookId] = False

    def handle_response_cancel_orders(self, simulation, payload):
        self.log(f"BOOK {payload.requestPayload.bookId} | CANCELLED ORDERS : {','.join([str(c.id) for c in payload.requestPayload.cancellations])}")
    
    def handle_error_response_cancel_orders(self, simulation, payload):
        if 'do not exist' not in payload.errorPayload.message:
            self.log(f"BOOK {payload.requestPayload.bookId} |  ERROR CANCELLING ORDER : {payload.errorPayload.message}")
    
    def handle_event_trade(self, simulation, payload):
        bookId = payload.bookId
        # NOTE Keep only last trade
        self.tradeEvent[bookId] = (float(payload.trade.price()),self.currentTimestamp)

        if self.debug:
            account = simulation.account(self.name())[bookId]
            self.log(f"BOOK {payload.bookId} | BASE : {self.wealthFrac*float(account.base.getFree())} | QUOTE : {self.wealthFrac*float(account.quote.getFree())}")
        # if self.tradeEvent[(self.currentTimestamp,bookId)] != self.priceHist[bookId][-1]:
        self.priceHist[bookId] = np.append(self.priceHist[bookId], self.tradeEvent[bookId][0])
        # self.priceHist[bookId] = self.priceHist[bookId][-int(self.Tinit * self.timeScale)-1:]
        self.ret[bookId] = np.append(self.ret[bookId],np.log(self.priceHist[bookId][-1]/self.priceHist[bookId][-2]))
        if len(self.ret[bookId]) > self.n_trades:
            self.ret[bookId] = self.ret[bookId][-self.n_trades:]
            self.priceHist[bookId] = self.priceHist[bookId][-self.n_trades:]
        if self.debug:
            self.log(' , '.join([str(p) for p in self.priceHist[bookId][-10:]]))

    
    def handle_event_simulation_stop(self, simulation, payload):
        self.log("-----The simulation ends now----")

    # ------------------------- Trading Logic ------------------------- #    
    def priceFcast(self, price, priceF, bookId):
        """
        Forecasts the asset price based on fundamental, chartist, and noise components.
        """
        np.random.seed()
        fundamental = np.log(priceF/price)
        noise = self.sigmaEps*np.random.randn()
        revrets = self.ret[bookId] #[-1:(-self.tau_agent-1):-1]
        if self.debug:
            self.log(f'Length of returns: {len(revrets)}')
            self.log(f'Returns: {revrets}')
        chartist = (1/self.tau_agent)*np.sum(revrets)
        self.Fcast = self.fcastAdjust * (
            (self.fundWeight / self.tauF) * fundamental + \
            self.chartWeight * chartist + \
            self.noiseWeight * noise
        )
        self.pFcast = price * np.exp(self.Fcast)
        self.var_returns = np.var(revrets)
        print(f'BOOK: {bookId} | AGENT: {self.name()} | VARIANCE OF RETURNS: {self.var_returns}')

    def log_inputs(self, simulation, price_star, price_trade, bid_highest, ask_lowest, price_m, price_M, bookId):
        self.log(f'BOOK {bookId} | {self.fcastAdjust * self.fundWeight=}')
        self.log(f'BOOK {bookId} | {self.fcastAdjust * self.chartWeight=}')
        self.log(f'BOOK {bookId} | {self.fcastAdjust * self.noiseWeight=}')
        self.log(f'BOOK {bookId} | {self.Fcast=}')
        self.log(f'BOOK {bookId} | {np.exp(self.Fcast)=}')
        self.log(f'BOOK {bookId} | {self.pFcast=}')
        self.log(f'BOOK {bookId} | {price_star=}')
        self.log(f'BOOK {bookId} | {price_trade=}')
        self.log(f'BOOK {bookId} | {bid_highest=}')
        self.log(f'BOOK {bookId} | {ask_lowest=}')
        self.log(f'BOOK {bookId} | {price_m=}')
        self.log(f'BOOK {bookId} | {price_M=}')
        self.log(f'BOOK {bookId} | {self.Stock_t[bookId]=}')
        self.log(f'BOOK {bookId} | {self.Cash_t[bookId]=}')
        self.log(f'BOOK {bookId} | {self.r_aversion_agent=}')
        self.log(f'BOOK {bookId} | {self.var_returns=}')
        self.log(f'BOOK {bookId} | {simulation.processValue("fundamental", bookId) =}')
        

    def placeorder_STA(self, simulation, ask_lowest, bid_highest, Timestamp, bookId):
        """
        Places an order based on the Chiarella trading strategy.
        """
        # Method for satisfaction price calculation.
        def eq_satisfaction(p):
            return (np.log(self.pFcast/p)/(self.r_aversion_agent*self.var_returns*p))-self.Stock_t[bookId]

        # Method for calculating minimum price, complying with agent's current position.
        def eq_smallest(p):
            return p*((np.log(self.pFcast/p)/(self.r_aversion_agent*self.var_returns*p))-self.Stock_t[bookId])-self.Cash_t[bookId]

        if self.r_aversion_agent*self.var_returns == 0:
            return {'OrderType': 'NO_ORDER'}
        
        price_star = fsolve(eq_satisfaction, 1)
        price_m = fsolve(eq_smallest, 1)
        price_M = self.pFcast

        if self.debug:
            self.log(f'Check satisfaction:{(np.log(self.pFcast/price_star)/(self.r_aversion_agent*self.var_returns*price_star))-self.Stock_t[bookId]}')
            self.log(f'Check smallest: {price_m*((np.log(self.pFcast/price_m)/(self.r_aversion_agent*self.var_returns*price_m))-self.Stock_t[bookId])-self.Cash_t[bookId]}')

        # Place an order if the calculations are accurate, otherwise no order.
        if np.isclose(eq_satisfaction(price_star), 0.0, atol=1e-05) and np.isclose(eq_smallest(price_m), 0.0, atol=1e-05):
            # The agent does not place any order if the condition below is not satisfied.
            price_star = price_star[0]
            price_m = price_m[0]
            price_trade = np.random.uniform(low=price_m, high=price_M)
            if price_m > 0 and price_m <= price_star and price_star <= price_M:
                pass
            else:
                self.log(f"BOOK {bookId} | CONDITIONS FOR UNIQUE SOLUTION ARE NOT FULFILLED")
                return {'OrderType': 'NO_ORDER'}

            # Adjusting the asset to the exchange's regulatory ticksize.
            np.random.seed()
            if self.debug:
                self.log_inputs(simulation, price_star, price_trade, bid_highest, ask_lowest, price_m, price_M, bookId)
            m_priceIncrement = 10**(-self.priceDecimals)
            if price_trade < price_star:
                if price_trade < ask_lowest:
                    bid_order_volume = np.log(self.pFcast/price_trade)/(self.r_aversion_agent*self.var_returns*price_trade)-self.Stock_t[bookId]
                else:
                    bid_order_volume = np.log(self.pFcast/ask_lowest)/(self.r_aversion_agent*self.var_returns*ask_lowest)-self.Stock_t[bookId]
                limit_bid_price = np.round(price_trade / m_priceIncrement) * m_priceIncrement
                if self.debug:
                    self.log(f'BUY LIMIT {bid_order_volume}@{limit_bid_price}')
                return {'OrderType': 'LIMIT_ORDER' , 'Price': limit_bid_price.item() , 'Volume': bid_order_volume.item(), 'OrderDirection':'Bid'}
            elif price_trade == price_star:
                return {'OrderType': 'NO_ORDER'}
            elif price_star < price_trade:
                if price_trade <= bid_highest:
                    sell_order_volume = self.Stock_t[bookId]-np.log(self.pFcast/bid_highest)/(self.r_aversion_agent*self.var_returns*bid_highest)
                else:
                    sell_order_volume = self.Stock_t[bookId]-np.log(self.pFcast/price_trade)/(self.r_aversion_agent*self.var_returns*price_trade)
                limit_sell_price = np.round(price_trade / m_priceIncrement) * m_priceIncrement
                if self.debug:
                    self.log(f'BOOK {bookId} | SELL LIMIT {sell_order_volume}@{limit_sell_price}')
                return {'OrderType': 'LIMIT_ORDER' , 'Price': limit_sell_price.item() , 'Volume': sell_order_volume.item(), 'OrderDirection':'Sell'}
        else:
            return {'OrderType': 'NO_ORDER'}


    def orderconvertsimulator_STA(self, simulation, Order, bookId, placeOrderLatency):
        """
        Converts the generated order to a format acceptable by the simulator.
        """
        m_volumeIncrement = 10**(-self.volumeDecimals)
        if Order['OrderType'] == "NO_ORDER" or Order['Volume'] < 0:
            return
        
        if Order['OrderDirection']== "Bid":
            if Order['OrderType'] == "LIMIT_ORDER":
                if Order['Volume'] > self.Cash_t[bookId] / Order['Price']:
                    Order['Volume'] = self.Cash_t[bookId] / Order['Price']
                if Order['Volume'] <= 0:
                    return
                Order['Volume'] = math.floor(Order['Volume']/m_volumeIncrement)*m_volumeIncrement
                self.orderFlag[bookId] = True
                LimitOrderPayload = PlaceOrderLimitPayload(OrderDirection.Buy, Decimal(Order['Volume']), Decimal(Order['Price']), Decimal(0), int(bookId))
                simulation.dispatchMessage(self.currentTimestamp, placeOrderLatency, self.name(), self.exchange, "PLACE_ORDER_LIMIT", LimitOrderPayload)
                self.log(f"BOOK {bookId} | SUBMITTED BUY LIMIT ORDER FOR {Order['Volume']}@{Order['Price']}")
            elif Order['OrderType'] == "MARKET_ORDER":
                if Order['Volume'] > self.Cash_t[bookId] / self.ask_lowest[bookId]:
                    Order['Volume'] =  self.Cash_t[bookId] / self.ask_lowest[bookId]
                if Order['Volume'] <= 0:
                    return
                Order['Volume'] = math.floor(Order['Volume']/m_volumeIncrement)*m_volumeIncrement
                self.orderFlag[bookId] = True
                MarketOrderPayload = PlaceOrderMarketPayload(OrderDirection.Buy, Decimal(Order['Volume']), Decimal(0), int(bookId))
                simulation.dispatchMessage(self.currentTimestamp, placeOrderLatency, self.name(), self.exchange, "PLACE_ORDER_MARKET", MarketOrderPayload)
                self.log(f"BOOK {bookId} | SUBMITTED BUY MARKET ORDER FOR {Order['Volume']}")
        elif Order['OrderDirection']== "Sell":
            if Order['Volume'] > self.Stock_t[bookId]:
                Order['Volume'] = self.Stock_t[bookId]
            if Order['Volume'] <= 0:
                return
            Order['Volume'] = math.floor(Order['Volume']/m_volumeIncrement)*m_volumeIncrement
            if Order['OrderType'] == "LIMIT_ORDER":
                self.orderFlag[bookId] = True
                LimitOrderPayload = PlaceOrderLimitPayload(OrderDirection.Sell, Decimal(Order['Volume']), Decimal(Order['Price']), Decimal(0), int(bookId))
                #LimitOrderPayload.bookId = bookId
                simulation.dispatchMessage(self.currentTimestamp, placeOrderLatency, self.name(), self.exchange, "PLACE_ORDER_LIMIT", LimitOrderPayload)
                self.log(f"BOOK {bookId} | SUBMITTED SELL LIMIT ORDER FOR {Order['Volume']}@{Order['Price']}")
            elif Order['OrderType'] == "MARKET_ORDER":
                self.orderFlag[bookId] = True
                MarketOrderPayload = PlaceOrderMarketPayload(OrderDirection.Sell, Decimal(Order['Volume']), Decimal(0), int(bookId))
                #MarketOrderPayload.bookId = bookId
                simulation.dispatchMessage(self.currentTimestamp, placeOrderLatency, self.name(), self.exchange, "PLACE_ORDER_MARKET", MarketOrderPayload)
                self.log(f"BOOK {bookId} | SUBMITTED SELL MARKET ORDER FOR {Order['Volume']}")

    # ------------------------- NB-ACD and Agent Selection ------------------------- #
    def getNextInterArrival(self, bookId, randomSeed, disableRandomSeed = False):
        """
        Computes the next interarrival duration using a Negative Binomial ACD model.
        """
        if disableRandomSeed:
            pass
        else:
            np.random.seed(randomSeed + bookId)
        while True:
            p_t = self.rACD[bookId] /(self.rACD[bookId] + self.psiACD[bookId])
            x_t = nbinom.rvs(self.rACD[bookId], p_t)
            self.psiACD[bookId] = self.omegaACD[bookId] + self.alphaACD[bookId] * x_t + self.betaACD[bookId] * self.psiACD[bookId]
            self.psiACD[bookId] = min(self.psiACD[bookId], 1/self.timeScale)
            if x_t > 0: 
                return x_t
            
            
    def simulateInterArrivals(self, T, r_hat, omega_hat, alpha_hat, beta_hat, timeScale, randomSeed):
        """
        Simulates interarrival durations using a NB-ACD(1,1) process.
        """
        np.random.seed(randomSeed)
        simulated_times = []
        psi_t = omega_hat  # Initialize with estimated ω
        while len(simulated_times) < T:
            p_t = r_hat / (r_hat + psi_t)  # Compute probability parameter
            x_t = nbinom.rvs(r_hat, p_t)   # Sample from Negative Binomial
            if x_t > 0: 
                simulated_times.append(x_t)
            psi_t = omega_hat + alpha_hat * x_t + beta_hat * psi_t
            psi_t = min(psi_t, 1/timeScale)
        # Convert to DataFrame
        simulated_data = pd.DataFrame({"simulated_duration": simulated_times})
        return (simulated_data['simulated_duration']).to_frame()
    def marketFeedLatency(self):
        # Gaussian latency
        return  min(abs(int(np.random.normal(self.mflMean,self.mflSTD))), int(self.mflMean+self.mflSTD*3))
    def orderPlacementLatency(self):
        # old OPL with Beta distribution, note wrong variable names
        # beta_sample = np.random.beta(self.alphaDelay, self.betaDelay, size=1)
        # latency = self.minFeedLatency + (self.maxFeedLatency-self.minFeedLatency) * beta_sample
        # return int(latency)
        ray_sample = rayleigh.rvs(scale=self.opLatencyScaleRay, size=1)
        latency = self.minOPLatency + (self.maxOPLatency-self.minOPLatency) * ray_sample
        latency = int(min(latency, self.maxOPLatency))
        return latency

    def decisionMakingDelay(self):
        return min(abs(int(np.random.normal(self.delayMean,self.delaySTD))),int(self.delayMean+self.delaySTD*3))
    def rayleigh_latencies_ns(self, scale=45_000_000, seed=None):
        """
        WIP: replacee the above
        """
        
        batch = rayleigh.rvs(scale=scale, size=1)
        shifted = batch + self.minOPLatency
        shifted = shifted if shifted <= self.maxOPLatency else self.maxOPLatency
        return shifted

    def agentIdSelector(self, bookId, randomSeed):
        """
        Determines the agent's ID based on its name and counts of different agent types.
        """
        matchID = re.search(r'_(\d+)$', self.name())
        if not matchID:
            raise ValueError("Agent name does not contain an ID")
        base_id = int(matchID.group(1))
        if self.name().startswith("STYLIZED"):
            agentID = base_id + self.cCount + self.fCount + self.hCount + self.iCount + self.nCount
            match = re.search(r'CHARTIST', self.name())
            if match:
                agentID = base_id  
            match = re.search(r'FUNDAMENTALIST', self.name())
            if match:
                agentID = base_id + self.cCount
            match = re.search(r'NOISE', self.name())
            if match:
                agentID = base_id + self.cCount + self.fCount + self.hCount + self.iCount
        if self.name().startswith("HIGH"):              
            agentID = base_id + self.cCount + self.fCount
                    
        wCT = np.full(self.cCount, self.activityC) 
        wFT = np.full(self.fCount, self.activityF)         
        wHFT = np.full(self.hCount, 0)
        wI = np.full(self.iCount, 0)
        wNT = np.full(self.nCount, self.activityN)
        wST = np.full(self.sCount, self.activityS)
        wSelection = np.concatenate((wCT, wFT, wHFT, wI, wNT, wST))
        wSelection = wSelection / np.sum(wSelection)

        np.random.seed(randomSeed)
        # Generate raw Rayleigh samples
        samples = rayleigh.rvs(scale=self.scaleR, size=1, random_state=randomSeed)

        # Map to integers [1, 10] based on value thresholds
        # We'll create bins such that smaller values are more likely to fall in lower bins
        bins = np.linspace(0, 5, 10)  # 9 bin edges → 10 bins
        digitized = np.digitize(samples, bins, right=False)  # values 1 through 10
        agentIDSel = np.random.choice(len(wSelection),size=digitized[0], p=wSelection)
        return agentIDSel, agentID
    
    def selectNextAgent(self, bookId, randomSeed):
        """
        Selects the next agent who will place an order.
        """
        # TODO From fear and gread index select more than one
        agentIDSel, agentID = self.agentIdSelector(bookId, randomSeed)
        return agentID in agentIDSel
    
    # ------------------------- Conversion of Time Scale ------------------------- #
    def timeScaleConverter(self, timeScale):
        scales = {
            "ns": 1e-9,
            "us": 1e-6,
            "ms": 1e-3,
            "s": 1
        }
        try:
            return scales[timeScale]
        except KeyError:
            raise ValueError("The specified time-scale has not been defined")
        
    # ------------------------- Scheduler of Dispatch RETRIEVE_L1 Message ------------------------- #
    def schedulerDispatchL1Message(self, simulation, bookId):
        if self.nextOrderArrival[bookId] - self.currentTimestamp < 2: 
            pass
        else:
            L1Payload = RetrieveL1Payload()
            L1Payload.bookId = bookId
            simulation.dispatchMessage(self.currentTimestamp, self.nextOrderArrival[bookId] - self.currentTimestamp - 1, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)
        if self.nextOrderArrival[bookId] - self.currentTimestamp < 3: 
            pass
        else:
            L1Payload = RetrieveL1Payload()
            L1Payload.bookId = bookId
            simulation.dispatchMessage(self.currentTimestamp, self.nextOrderArrival[bookId] - self.currentTimestamp - 2, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)





