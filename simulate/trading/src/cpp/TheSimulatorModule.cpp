/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"

#include <fmt/format.h>

#include <pybind11/embed.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

//-------------------------------------------------------------------------

namespace py = pybind11;

using namespace taosim;

//-------------------------------------------------------------------------

PYBIND11_EMBEDDED_MODULE(thesimulator, m)
{
    py::class_<Simulation>(m, "Simulation")
        .def("logDir", [](Simulation& self) { return self.logDir().string(); })
        .def("duration", &Simulation::duration)
        .def("currentTimestamp", &Simulation::currentTimestamp)
        .def("dispatchGenericMessage", &Simulation::dispatchGenericMessage)
        .def("dispatchMessage", &Simulation::dispatchMessage)
        .def("queueMessage", &Simulation::queueMessage)
        .def("account", &Simulation::account)
        .def("bookCount", [](Simulation& self) { return self.exchange()->books().size(); })
        .def("processValue", [](Simulation& self, const std::string& name, BookId bookId) {
            return self.exchange()->process(name, bookId)->value();
        })
        .def("getAgentId", [](Simulation& self, const std::string& name) -> AgentId {
            const auto& map = self.exchange()->accounts().idBimap().left;
            auto it = map.find(name);
            if (it == map.end()) [[unlikely]] {
                throw std::runtime_error{fmt::format(
                    "{}: No agent with name '{}' found",
                    std::source_location::current().function_name(), name)};
            }
            return it->second;
        })
        ;

    py::class_<decimal_t>(m, "Decimal")
        .def(py::init<double>())
        .def("__float__", [](decimal_t self) { return util::decimal2double(self); })
        .def("__str__", [](decimal_t self) { return fmt::format("{}", self); })
        .def("__add__", [](decimal_t a, decimal_t b) { return a + b; })
        .def("__sub__", [](decimal_t a, decimal_t b) { return a - b; })
        .def("__mul__", [](decimal_t a, decimal_t b) { return a * b; })
        .def("__truediv__", [](decimal_t a, decimal_t b) { return a / b; })
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        ;

    py::class_<accounting::Balance>(m, "Balance")
        .def("getFree", [](const accounting::Balance& self) { return self.getFree(); })
        .def("getTotal", [](const accounting::Balance& self) { return self.getTotal(); })
        .def("getReserved", [](const accounting::Balance& self) { return self.getReserved(); })
        ;

    py::class_<accounting::Balances>(m, "Balances")
        .def_readonly("base", &accounting::Balances::base)
        .def_readonly("quote", &accounting::Balances::quote)
        .def("getLeverage", &accounting::Balances::getLeverage)
        .def("getWealth", &accounting::Balances::getWealth)
        .def("getReservationInQuote", &accounting::Balances::getReservationInQuote)
        .def("getReservationInBase", &accounting::Balances::getReservationInBase)
        ;

    py::class_<accounting::Account>(m, "Account")
        .def(
            "__getitem__",
            [](const accounting::Account& self, BookId bookId) { return self.at(bookId); },
            py::return_value_policy::reference)
        ;

    py::enum_<OrderDirection>(m, "OrderDirection")
        .value("Buy", OrderDirection::BUY)
        .value("Sell", OrderDirection::SELL)
        ;

    py::class_<MessagePayload, std::shared_ptr<MessagePayload>>(m, "MessagePayload")
        ;

    py::class_<ErrorResponsePayload, std::shared_ptr<ErrorResponsePayload>>(
        m, "ErrorResponsePayload")
        .def_readwrite("message", &ErrorResponsePayload::message)
        ;

    py::class_<SuccessResponsePayload, std::shared_ptr<SuccessResponsePayload>>(
        m, "SuccessResponsePayload")
        .def_readwrite("message", &SuccessResponsePayload::message)
        ;

    py::class_<EmptyPayload, MessagePayload, std::shared_ptr<EmptyPayload>>(m, "EmptyPayload")
        .def(py::init<>())
        ;

    py::class_<PlaceOrderMarketPayload, MessagePayload, std::shared_ptr<PlaceOrderMarketPayload>>(
        m, "PlaceOrderMarketPayload")
        .def(py::init<OrderDirection, decimal_t, decimal_t, BookId, Currency>())
        .def_readwrite("direction", &PlaceOrderMarketPayload::direction)
        .def_readwrite("volume", &PlaceOrderMarketPayload::volume)
        .def_readwrite("leverage", &PlaceOrderMarketPayload::leverage)
        .def_readwrite("bookId", &PlaceOrderMarketPayload::bookId)
        .def_readwrite("currency", &PlaceOrderMarketPayload::currency)
        .def_readwrite("clientOrderId", &PlaceOrderMarketPayload::clientOrderId)
        ;

    py::class_<
        PlaceOrderMarketResponsePayload,
        MessagePayload,
        std::shared_ptr<PlaceOrderMarketResponsePayload>>(m, "PlaceOrderMarketResponsePayload")
        .def(py::init<OrderID, const std::shared_ptr<PlaceOrderMarketPayload>&>())
        .def_readwrite("id", &PlaceOrderMarketResponsePayload::id)
        .def_readwrite("requestPayload", &PlaceOrderMarketResponsePayload::requestPayload)
        ;

    py::class_<
        PlaceOrderMarketErrorResponsePayload,
        MessagePayload,
        std::shared_ptr<PlaceOrderMarketErrorResponsePayload>>(
        m, "PlaceOrderMarketErrorResponsePayload")
        .def(py::init<PlaceOrderMarketPayload::Ptr, ErrorResponsePayload::Ptr>())
        .def_readwrite("requestPayload", &PlaceOrderMarketErrorResponsePayload::requestPayload)
        .def_readwrite("errorPayload", &PlaceOrderMarketErrorResponsePayload::errorPayload)
        ;

    py::class_<PlaceOrderLimitPayload, MessagePayload, std::shared_ptr<PlaceOrderLimitPayload>>(
        m, "PlaceOrderLimitPayload")
        .def(py::init<OrderDirection, decimal_t, decimal_t, decimal_t, BookId, Currency>())
        .def_readwrite("direction", &PlaceOrderLimitPayload::direction)
        .def_readwrite("volume", &PlaceOrderLimitPayload::volume)
        .def_readwrite("price", &PlaceOrderLimitPayload::price)
        .def_readwrite("leverage", &PlaceOrderLimitPayload::leverage)
        .def_readwrite("bookId", &PlaceOrderLimitPayload::bookId)
        .def_readwrite("currency", &PlaceOrderLimitPayload::currency)
        .def_readwrite("clientOrderId", &PlaceOrderLimitPayload::clientOrderId)
        ;

    py::class_<
        PlaceOrderLimitResponsePayload,
        MessagePayload,
        std::shared_ptr<PlaceOrderLimitResponsePayload>>(m, "PlaceOrderLimitResponsePayload")
        .def(py::init<OrderID, const std::shared_ptr<PlaceOrderLimitPayload>&>())
        .def_readwrite("id", &PlaceOrderLimitResponsePayload::id)
        .def_readwrite("requestPayload", &PlaceOrderLimitResponsePayload::requestPayload)
        ;

    py::class_<
        PlaceOrderLimitErrorResponsePayload,
        MessagePayload,
        std::shared_ptr<PlaceOrderLimitErrorResponsePayload>>(
        m, "PlaceOrderLimitErrorResponsePayload")
        .def(py::init<PlaceOrderLimitPayload::Ptr, ErrorResponsePayload::Ptr>())
        .def_readwrite("requestPayload", &PlaceOrderLimitErrorResponsePayload::requestPayload)
        .def_readwrite("errorPayload", &PlaceOrderLimitErrorResponsePayload::errorPayload)
        ;

    py::class_<RetrieveOrdersPayload, MessagePayload, std::shared_ptr<RetrieveOrdersPayload>>(
        m, "RetrieveOrdersPayload")
        .def(py::init<std::vector<OrderID>, BookId>())
        .def_readwrite("ids", &RetrieveOrdersPayload::ids)
        .def_readwrite("bookId", &RetrieveOrdersPayload::bookId)
        ;

    py::class_<
        RetrieveOrdersResponsePayload,
        MessagePayload,
        std::shared_ptr<RetrieveOrdersResponsePayload>>(m, "RetrieveOrdersResponsePayload")
        .def(py::init<>())
        .def_readwrite("orders", &RetrieveOrdersResponsePayload::orders)
        .def_readwrite("bookId", &RetrieveOrdersResponsePayload::bookId)
        ;

    py::class_<taosim::event::Cancellation>(m, "Cancellation")
        .def(py::init<OrderID, std::optional<decimal_t>>())
        .def_readwrite("id", &taosim::event::Cancellation::id)
        .def_readwrite("volume", &taosim::event::Cancellation::volume)
        ;

    py::class_<CancelOrdersPayload, MessagePayload, std::shared_ptr<CancelOrdersPayload>>(
        m, "CancelOrdersPayload")
        .def(py::init<std::vector<taosim::event::Cancellation>, BookId>())
        .def_readwrite("cancellations", &CancelOrdersPayload::cancellations)
        .def_readwrite("bookId", &CancelOrdersPayload::bookId)
        ;

    py::class_<
        CancelOrdersResponsePayload,
        MessagePayload,
        std::shared_ptr<CancelOrdersResponsePayload>>(m, "CancelOrdersResponsePayload")
        .def(py::init<std::vector<OrderID>, const std::shared_ptr<CancelOrdersPayload>&>())
        .def_readwrite("orderIds", &CancelOrdersResponsePayload::orderIds)
        .def_readwrite("requestPayload", &CancelOrdersResponsePayload::requestPayload)
        ;

    py::class_<
        CancelOrdersErrorResponsePayload,
        MessagePayload,
        std::shared_ptr<CancelOrdersErrorResponsePayload>>(
        m, "CancelOrdersErrorResponsePayload")
        .def(py::init<std::vector<OrderID>, CancelOrdersPayload::Ptr, ErrorResponsePayload::Ptr>())
        .def_readwrite("orderIds", &CancelOrdersErrorResponsePayload::orderIds)
        .def_readwrite("requestPayload", &CancelOrdersErrorResponsePayload::requestPayload)
        .def_readwrite("errorPayload", &CancelOrdersErrorResponsePayload::errorPayload)
        ;

    py::class_<RetrieveL1Payload, MessagePayload, RetrieveL1Payload::Ptr>(m, "RetrieveL1Payload")
        .def(py::init<>())
        .def_readwrite("bookId", &RetrieveL1Payload::bookId)
        ;

    py::class_<RetrieveL1ResponsePayload, MessagePayload, RetrieveL1ResponsePayload::Ptr>(
        m, "RetrieveL1ResponsePayload")
        .def(
            py::init<
                Timestamp,
                decimal_t,
                decimal_t,
                decimal_t,
                decimal_t,
                decimal_t,
                decimal_t,
                BookId>())
        .def_readwrite("time", &RetrieveL1ResponsePayload::time)
        .def_readwrite("bestAskPrice", &RetrieveL1ResponsePayload::bestAskPrice)
        .def_readwrite("bestAskVolume", &RetrieveL1ResponsePayload::bestAskVolume)
        .def_readwrite("askTotalVolume", &RetrieveL1ResponsePayload::askTotalVolume)
        .def_readwrite("bestBidPrice", &RetrieveL1ResponsePayload::bestBidPrice)
        .def_readwrite("bestBidVolume", &RetrieveL1ResponsePayload::bestBidVolume)
        .def_readwrite("bidTotalVolume", &RetrieveL1ResponsePayload::bidTotalVolume)
        .def_readwrite("bookId", &RetrieveL1ResponsePayload::bookId)
        ;

    py::class_<taosim::book::BookTradeStats>(m, "BookTradeStats")
        .def(py::init<>())
        .def_readwrite("tradeCount", &taosim::book::BookTradeStats::tradeCount)
        .def_readwrite("volumeSum", &taosim::book::BookTradeStats::volumeSum)
        .def_readwrite("notionalSum", &taosim::book::BookTradeStats::notionalSum)
        .def_readwrite("logReturnSum", &taosim::book::BookTradeStats::logReturnSum)
        .def_readwrite("logReturnSqSum", &taosim::book::BookTradeStats::logReturnSqSum)
        .def_readwrite("lastTradePrice", &taosim::book::BookTradeStats::lastTradePrice)
        .def_readwrite("lastTradeTime", &taosim::book::BookTradeStats::lastTradeTime)
        ;

    py::class_<RetrieveL1ExtPayload, MessagePayload, RetrieveL1ExtPayload::Ptr>(
        m, "RetrieveL1ExtPayload")
        .def(py::init<>())
        .def_readwrite("bookId", &RetrieveL1ExtPayload::bookId)
        ;

    py::class_<RetrieveL1ExtResponsePayload, MessagePayload, RetrieveL1ExtResponsePayload::Ptr>(
        m, "RetrieveL1ExtResponsePayload")
        .def(
            py::init<
                Timestamp,
                decimal_t,
                decimal_t,
                decimal_t,
                decimal_t,
                decimal_t,
                decimal_t,
                taosim::book::BookTradeStats,
                BookId>())
        .def_readwrite("time", &RetrieveL1ExtResponsePayload::time)
        .def_readwrite("bestAskPrice", &RetrieveL1ExtResponsePayload::bestAskPrice)
        .def_readwrite("bestAskVolume", &RetrieveL1ExtResponsePayload::bestAskVolume)
        .def_readwrite("askTotalVolume", &RetrieveL1ExtResponsePayload::askTotalVolume)
        .def_readwrite("bestBidPrice", &RetrieveL1ExtResponsePayload::bestBidPrice)
        .def_readwrite("bestBidVolume", &RetrieveL1ExtResponsePayload::bestBidVolume)
        .def_readwrite("bidTotalVolume", &RetrieveL1ExtResponsePayload::bidTotalVolume)
        .def_readwrite("tradeStats", &RetrieveL1ExtResponsePayload::tradeStats)
        .def_readwrite("bookId", &RetrieveL1ExtResponsePayload::bookId)
        ;

    py::class_<
        SubscribeEventTradeByOrderPayload,
        MessagePayload,
        std::shared_ptr<SubscribeEventTradeByOrderPayload>>(m, "SubscribeEventTradeByOrderPayload")
        .def(py::init<OrderID>())
        .def_readwrite("id", &SubscribeEventTradeByOrderPayload::id)
        ;

    py::class_<EventOrderMarketPayload, MessagePayload, std::shared_ptr<EventOrderMarketPayload>>(
        m, "EventOrderMarketPayload")
        .def(py::init<MarketOrder>())
        .def_readonly("order", &EventOrderMarketPayload::order)
        .def_readonly("bookId", &EventOrderMarketPayload::bookId)
        .def_readonly("agentId", &EventOrderMarketPayload::agentId)
        ;

    py::class_<EventOrderLimitPayload, MessagePayload, std::shared_ptr<EventOrderLimitPayload>>(
        m, "EventOrderLimitPayload")
        .def(py::init<LimitOrder>())
        .def_readonly("order", &EventOrderLimitPayload::order)
        .def_readonly("bookId", &EventOrderLimitPayload::bookId)
        .def_readonly("agentId", &EventOrderLimitPayload::agentId)
        ;

    py::class_<Trade, Trade::Ptr>(m, "Trade")
        .def("id", [](const Trade& self) { return self.id(); })
        .def("direction", [](const Trade& self) { return self.direction(); })
        .def("timestamp", [](const Trade& self) { return self.timestamp(); })
        .def("aggressingOrderId", [](const Trade& self) { return self.aggressingOrderID(); })
        .def("restingOrderId", [](const Trade& self) { return self.restingOrderID(); })
        .def("volume", [](const Trade& self) { return self.volume(); })
        .def("price", [](const Trade& self) { return self.price(); })
        ;

    py::class_<EventTradePayload, MessagePayload, std::shared_ptr<EventTradePayload>>(
        m, "EventTradePayload")
        .def(py::init<Trade,TradeLogContext, BookId>())
        .def_readonly("trade", &EventTradePayload::trade)
        .def_readonly("context", &EventTradePayload::context)
        .def_readonly("bookId", &EventTradePayload::bookId)
        .def_readonly("clientOrderId", &EventTradePayload::clientOrderId)
        ;

    py::class_<TradeLogContext, TradeLogContext::Ptr>(m, "TradeLogContext")
        .def_readonly("aggressingAgentId", &TradeLogContext::aggressingAgentId)
        .def_readonly("restingAgentId", &TradeLogContext::restingAgentId)
        ;
}

//-------------------------------------------------------------------------