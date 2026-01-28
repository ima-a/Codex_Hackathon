import { useState, useEffect } from 'react';
import ComparisonChart from '../components/ComparisonChart';

type Scenario = 'normal' | 'spike' | 'drop';

interface Recommendation {
    orderQuantity: number;
    reason: string;
    expectedServiceLevel: number;
    stockoutRisk: 'LOW' | 'MEDIUM' | 'HIGH';
    estimatedCost: number;
}

const InventoryManager = () => {
    const [selectedStore, setSelectedStore] = useState('store_1');
    const [selectedSKU, setSelectedSKU] = useState('SKU_001');
    const [scenario, setScenario] = useState<Scenario>('normal');
    const [whatIfQuantity, setWhatIfQuantity] = useState('');
    const [recommendation, setRecommendation] = useState<Recommendation | null>(null);

    // Dynamic inventory data based on store and SKU
    const getInventoryData = () => {
        const storeData: Record<string, Record<string, any>> = {
            store_1: {
                SKU_001: { inventory: 200, reorderPoint: 150, demandForecast: 25, safetyStock: 50, inTransit: 100 },
                SKU_002: { inventory: 450, reorderPoint: 300, demandForecast: 45, safetyStock: 80, inTransit: 150 },
                SKU_003: { inventory: 120, reorderPoint: 100, demandForecast: 15, safetyStock: 30, inTransit: 50 },
            },
            store_2: {
                SKU_001: { inventory: 350, reorderPoint: 200, demandForecast: 35, safetyStock: 60, inTransit: 120 },
                SKU_002: { inventory: 280, reorderPoint: 250, demandForecast: 40, safetyStock: 70, inTransit: 130 },
                SKU_003: { inventory: 90, reorderPoint: 80, demandForecast: 12, safetyStock: 25, inTransit: 40 },
            },
            store_3: {
                SKU_001: { inventory: 180, reorderPoint: 120, demandForecast: 20, safetyStock: 40, inTransit: 80 },
                SKU_002: { inventory: 520, reorderPoint: 350, demandForecast: 50, safetyStock: 90, inTransit: 180 },
                SKU_003: { inventory: 150, reorderPoint: 110, demandForecast: 18, safetyStock: 35, inTransit: 60 },
            },
        };

        const data = storeData[selectedStore]?.[selectedSKU] || storeData.store_1.SKU_001;
        const daysOfSupply = Math.floor(data.inventory / Math.max(data.demandForecast, 1)); // Avoid divide by zero

        return {
            inventory: data.inventory,
            reorderPoint: data.reorderPoint,
            daysOfSupply: daysOfSupply,
            demandForecast: data.demandForecast,
            safetyStock: data.safetyStock,
            inTransit: data.inTransit,
        };
    };

    const currentStatus = getInventoryData();

    // Calculate stockout risk based on inventory position
    const calculateStockoutRisk = (inventory: number, demand: number, scenario: Scenario): 'LOW' | 'MEDIUM' | 'HIGH' => {
        const adjustedDemand = scenario === 'spike' ? demand * 1.8 : scenario === 'drop' ? demand * 0.7 : demand;
        const daysOfCoverage = inventory / Math.max(adjustedDemand, 1);

        if (daysOfCoverage < 5) return 'HIGH';
        if (daysOfCoverage < 10) return 'MEDIUM';
        return 'LOW';
    };

    // Calculate realistic service level based on inventory coverage
    const calculateServiceLevel = (orderQty: number, demand: number, inventory: number, scenario: Scenario): number => {
        const adjustedDemand = scenario === 'spike' ? demand * 1.8 : scenario === 'drop' ? demand * 0.7 : demand;
        const totalInventory = inventory + orderQty;
        const coverage = totalInventory / Math.max(adjustedDemand * 14, 1); // 14-day coverage target

        // Higher demand = harder to maintain service level
        const demandPenalty = Math.min(demand / 100, 0.05); // Max 5% penalty for high demand
        const baseServiceLevel = Math.min(0.99, 0.85 + coverage * 0.08) - demandPenalty;

        return Math.max(0.80, Math.min(0.99, baseServiceLevel));
    };

    // Generate dynamic reason text
    const generateReason = (inventory: number, reorderPoint: number, demand: number, scenario: Scenario): string => {
        const percentOfROP = (inventory / Math.max(reorderPoint, 1)) * 100;
        const daysOfSupply = Math.floor(inventory / Math.max(demand, 1));

        if (scenario === 'spike') {
            return `Demand spike predicted (${Math.floor(demand * 1.8)} units/day). Order urgently to prevent stockouts.`;
        }

        if (scenario === 'drop') {
            return `Demand declining to ${Math.floor(demand * 0.7)} units/day. Reduce order quantity to minimize holding costs.`;
        }

        // Normal scenario - dynamic based on actual inventory
        if (percentOfROP < 90) {
            return `⚠️ URGENT: Inventory at ${percentOfROP.toFixed(0)}% of reorder point. Only ${daysOfSupply} days of supply remaining.`;
        } else if (percentOfROP < 110) {
            return `Inventory approaching reorder point (${percentOfROP.toFixed(0)}% of ROP). Recommended to order ${Math.floor(demand * 14)} units for 14-day coverage.`;
        } else {
            return `Inventory healthy at ${percentOfROP.toFixed(0)}% of ROP. Maintain ${demand} units/day demand with ${daysOfSupply} days supply.`;
        }
    };

    // Generate recommendation - NOW CALLS RL MODEL API! 🚀
    useEffect(() => {
        const fetchRecommendation = async () => {
            try {
                // Call Flask API to get RL model prediction
                const apiUrl = `http://localhost:5000/api/recommend?store=${selectedStore}&sku=${selectedSKU}&scenario=${scenario}`;
                const response = await fetch(apiUrl);

                if (response.ok) {
                    const data = await response.json();

                    // Set recommendation from RL model
                    setRecommendation({
                        orderQuantity: data.orderQuantity,
                        reason: data.reason,  // Will show 🤖 if from RL model!
                        expectedServiceLevel: data.expectedServiceLevel,
                        stockoutRisk: data.stockoutRisk as 'LOW' | 'MEDIUM' | 'HIGH',
                        estimatedCost: data.estimatedCost,
                    });
                } else {
                    throw new Error('API unavailable');
                }
            } catch (error) {
                console.warn('⚠️ RL API unavailable, using fallback calculation:', error);

                // Fallback to client-side calculation if API is down
                const baseOrder = Math.floor(currentStatus.demandForecast * 14);
                const stockoutRisk = calculateStockoutRisk(currentStatus.inventory, currentStatus.demandForecast, scenario);

                const recommendations: Record<Scenario, Recommendation> = {
                    normal: {
                        orderQuantity: Math.max(0, baseOrder),
                        reason: '📊 Fallback: ' + generateReason(currentStatus.inventory, currentStatus.reorderPoint, currentStatus.demandForecast, 'normal'),
                        expectedServiceLevel: calculateServiceLevel(baseOrder, currentStatus.demandForecast, currentStatus.inventory, 'normal'),
                        stockoutRisk: stockoutRisk,
                        estimatedCost: Math.floor(baseOrder * 3.5),
                    },
                    spike: {
                        orderQuantity: Math.max(0, Math.floor(baseOrder * 1.5)),
                        reason: '📊 Fallback: ' + generateReason(currentStatus.inventory, currentStatus.reorderPoint, currentStatus.demandForecast, 'spike'),
                        expectedServiceLevel: calculateServiceLevel(Math.floor(baseOrder * 1.5), currentStatus.demandForecast, currentStatus.inventory, 'spike'),
                        stockoutRisk: calculateStockoutRisk(currentStatus.inventory, currentStatus.demandForecast, 'spike'),
                        estimatedCost: Math.floor(baseOrder * 1.5 * 3.5),
                    },
                    drop: {
                        orderQuantity: Math.max(0, Math.floor(baseOrder * 0.6)),
                        reason: '📊 Fallback: ' + generateReason(currentStatus.inventory, currentStatus.reorderPoint, currentStatus.demandForecast, 'drop'),
                        expectedServiceLevel: calculateServiceLevel(Math.floor(baseOrder * 0.6), currentStatus.demandForecast, currentStatus.inventory, 'drop'),
                        stockoutRisk: calculateStockoutRisk(currentStatus.inventory, currentStatus.demandForecast, 'drop'),
                        estimatedCost: Math.floor(baseOrder * 0.6 * 3.5),
                    },
                };

                setRecommendation(recommendations[scenario]);
            }
        };

        fetchRecommendation();
    }, [scenario, selectedStore, selectedSKU, currentStatus.demandForecast, currentStatus.inventory, currentStatus.reorderPoint]);

    // Calculate what-if prediction with scenario-adjusted demand + validation
    const calculateWhatIf = (quantity: number) => {
        // Input validation
        if (isNaN(quantity) || quantity < 0) {
            return null;
        }

        if (quantity > 10000) {
            // Cap at reasonable max
            quantity = 10000;
        }

        // Adjust demand based on scenario
        let adjustedDemand = currentStatus.demandForecast;
        if (scenario === 'spike') {
            adjustedDemand = currentStatus.demandForecast * 1.8;
        } else if (scenario === 'drop') {
            adjustedDemand = currentStatus.demandForecast * 0.7;
        }

        const totalInventory = currentStatus.inventory + quantity;
        const daysUntilStockout = Math.floor(totalInventory / Math.max(adjustedDemand, 1));
        const serviceLevel = calculateServiceLevel(quantity, currentStatus.demandForecast, currentStatus.inventory, scenario);
        const avgInventory = (currentStatus.inventory + quantity) / 2;
        const turnover = avgInventory > 0 ? (adjustedDemand * 30) / avgInventory : 0;

        return {
            serviceLevel: Math.min(0.99, Math.max(0.80, serviceLevel)),
            daysUntilStockout: Math.max(0, daysUntilStockout),
            totalCost: quantity * 3.5,
            turnover: Math.max(0, Math.min(20, turnover)), // Cap turnover at reasonable max
        };
    };

    // Re-calculate what-if when scenario changes
    const whatIfPrediction = whatIfQuantity && !isNaN(parseInt(whatIfQuantity)) && parseInt(whatIfQuantity) >= 0
        ? calculateWhatIf(parseInt(whatIfQuantity))
        : null;

    // Demand forecast data (7 days historical + 7 days predicted)
    const baseDemand = currentStatus.demandForecast;
    const demandForecastData = [
        { name: 'Day -6', actual: Math.max(0, baseDemand - 3), forecast: null },
        { name: 'Day -5', actual: Math.max(0, baseDemand + 3), forecast: null },
        { name: 'Day -4', actual: Math.max(0, baseDemand - 1), forecast: null },
        { name: 'Day -3', actual: Math.max(0, baseDemand + 1), forecast: null },
        { name: 'Day -2', actual: Math.max(0, baseDemand - 2), forecast: null },
        { name: 'Day -1', actual: Math.max(0, baseDemand), forecast: null },
        { name: 'Today', actual: Math.max(0, baseDemand + 2), forecast: Math.max(0, baseDemand + 2) },
        { name: 'Day +1', actual: null, forecast: Math.max(0, scenario === 'spike' ? baseDemand * 1.8 : scenario === 'drop' ? baseDemand * 0.7 : baseDemand + 1) },
        { name: 'Day +2', actual: null, forecast: Math.max(0, scenario === 'spike' ? baseDemand * 2 : scenario === 'drop' ? baseDemand * 0.6 : baseDemand) },
        { name: 'Day +3', actual: null, forecast: Math.max(0, scenario === 'spike' ? baseDemand * 1.9 : scenario === 'drop' ? baseDemand * 0.6 : baseDemand + 2) },
        { name: 'Day +4', actual: null, forecast: Math.max(0, scenario === 'spike' ? baseDemand * 1.4 : scenario === 'drop' ? baseDemand * 0.7 : baseDemand - 1) },
        { name: 'Day +5', actual: null, forecast: Math.max(0, scenario === 'spike' ? baseDemand * 1.2 : scenario === 'drop' ? baseDemand * 0.8 : baseDemand + 1) },
        { name: 'Day +6', actual: null, forecast: Math.max(0, scenario === 'spike' ? baseDemand * 1.1 : scenario === 'drop' ? baseDemand * 0.8 : baseDemand) },
    ];

    const applyRecommendation = () => {
        if (!recommendation) return;
        alert(`Order of ${recommendation.orderQuantity} units has been recommended for ${selectedStore} - ${selectedSKU}.\n\nExpected Service Level: ${(recommendation.expectedServiceLevel * 100).toFixed(1)}%\nStockout Risk: ${recommendation.stockoutRisk}\n\nIn a real system, this would integrate with your ordering API.`);
    };

    // Handle invalid what-if input
    const handleWhatIfChange = (value: string) => {
        // Allow empty string or valid numbers
        if (value === '' || /^\d+$/.test(value)) {
            setWhatIfQuantity(value);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-800">Inventory Manager</h1>
                    <p className="text-gray-600 mt-2">
                        AI-powered inventory recommendations for optimal stock levels
                    </p>
                </div>

                {/* Store & SKU Selector */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-6 border border-gray-200">
                    <div className="grid md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Select Store
                            </label>
                            <select
                                value={selectedStore}
                                onChange={(e) => setSelectedStore(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value="store_1">Store 1 - Downtown</option>
                                <option value="store_2">Store 2 - Mall</option>
                                <option value="store_3">Store 3 - Suburb</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Select SKU
                            </label>
                            <select
                                value={selectedSKU}
                                onChange={(e) => setSelectedSKU(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value="SKU_001">SKU_001 - Premium Widget</option>
                                <option value="SKU_002">SKU_002 - Standard Widget</option>
                                <option value="SKU_003">SKU_003 - Budget Widget</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Current Inventory
                            </label>
                            <div className="px-3 py-2 bg-gray-100 rounded-md">
                                <span className="text-2xl font-bold text-gray-800">{currentStatus.inventory}</span>
                                <span className="text-gray-600 ml-2">units</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* AI Recommendation Card - PROMINENT */}
                {recommendation && (
                    <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg shadow-lg p-8 mb-6 text-white">
                        <div className="flex items-start justify-between">
                            <div className="flex-1">
                                <div className="flex items-center mb-4">
                                    <svg className="w-8 h-8 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                    </svg>
                                    <h2 className="text-2xl font-bold">AI Recommendation</h2>
                                </div>

                                <div className="mb-4">
                                    <div className="text-5xl font-extrabold mb-2">
                                        ORDER NOW: {recommendation.orderQuantity} units
                                    </div>
                                    <p className="text-blue-100 text-lg">{recommendation.reason}</p>
                                </div>

                                <div className="grid grid-cols-3 gap-4 mb-6">
                                    <div className="bg-white bg-opacity-20 rounded-lg p-3">
                                        <p className="text-sm text-blue-100">Expected Service Level</p>
                                        <p className="text-2xl font-bold">{(recommendation.expectedServiceLevel * 100).toFixed(1)}%</p>
                                    </div>
                                    <div className="bg-white bg-opacity-20 rounded-lg p-3">
                                        <p className="text-sm text-blue-100">Stockout Risk</p>
                                        <p className={`text-2xl font-bold ${recommendation.stockoutRisk === 'HIGH' ? 'text-red-300' :
                                            recommendation.stockoutRisk === 'MEDIUM' ? 'text-yellow-300' : ''
                                            }`}>
                                            {recommendation.stockoutRisk}
                                        </p>
                                    </div>
                                    <div className="bg-white bg-opacity-20 rounded-lg p-3">
                                        <p className="text-sm text-blue-100">Estimated Cost</p>
                                        <p className="text-2xl font-bold">${recommendation.estimatedCost.toLocaleString()}</p>
                                    </div>
                                </div>

                                <button
                                    onClick={applyRecommendation}
                                    className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors shadow-md"
                                >
                                    Apply Recommendation
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Scenario Toggle */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-6 border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Demand Scenario</h3>
                    <div className="flex gap-4">
                        {(['normal', 'spike', 'drop'] as Scenario[]).map((sc) => (
                            <button
                                key={sc}
                                onClick={() => setScenario(sc)}
                                className={`px-6 py-3 rounded-lg font-medium transition-colors ${scenario === sc
                                    ? 'bg-blue-600 text-white shadow-md'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                            >
                                {sc.charAt(0).toUpperCase() + sc.slice(1)} Demand
                            </button>
                        ))}
                    </div>
                    <p className="text-sm text-gray-600 mt-3">
                        Toggle scenarios to see how AI recommendations change with different demand patterns
                    </p>
                </div>

                <div className="grid lg:grid-cols-2 gap-6 mb-6">
                    {/* Current Status */}
                    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">Current Status</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="border-l-4 border-blue-500 pl-3">
                                <p className="text-sm text-gray-600">Current Inventory</p>
                                <p className="text-2xl font-bold text-gray-800">{currentStatus.inventory} units</p>
                            </div>
                            <div className="border-l-4 border-green-500 pl-3">
                                <p className="text-sm text-gray-600">Reorder Point</p>
                                <p className="text-2xl font-bold text-gray-800">{currentStatus.reorderPoint} units</p>
                            </div>
                            <div className="border-l-4 border-yellow-500 pl-3">
                                <p className="text-sm text-gray-600">Days of Supply</p>
                                <p className="text-2xl font-bold text-gray-800">{currentStatus.daysOfSupply} days</p>
                            </div>
                            <div className="border-l-4 border-purple-500 pl-3">
                                <p className="text-sm text-gray-600">Demand Forecast</p>
                                <p className="text-2xl font-bold text-gray-800">{currentStatus.demandForecast} units/day</p>
                            </div>
                            <div className="border-l-4 border-indigo-500 pl-3">
                                <p className="text-sm text-gray-600">Safety Stock</p>
                                <p className="text-2xl font-bold text-gray-800">{currentStatus.safetyStock} units</p>
                            </div>
                            <div className="border-l-4 border-pink-500 pl-3">
                                <p className="text-sm text-gray-600">In-Transit</p>
                                <p className="text-2xl font-bold text-gray-800">{currentStatus.inTransit} units</p>
                            </div>
                        </div>
                    </div>

                    {/* What-If Simulator */}
                    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">What-If Simulator</h3>
                        <div className="mb-4">
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                What if I order:
                            </label>
                            <input
                                type="number"
                                min="0"
                                max="10000"
                                value={whatIfQuantity}
                                onChange={(e) => handleWhatIfChange(e.target.value)}
                                placeholder="Enter quantity (0-10000)..."
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {whatIfPrediction ? (
                            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                                <h4 className="font-semibold text-gray-800 mb-2">Predicted Results:</h4>
                                <div className="flex justify-between items-center">
                                    <span className="text-gray-700">Service Level:</span>
                                    <span className="font-bold text-green-600">{(whatIfPrediction.serviceLevel * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-gray-700">Days Until Stockout:</span>
                                    <span className="font-bold text-blue-600">{whatIfPrediction.daysUntilStockout} days</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-gray-700">Total Cost:</span>
                                    <span className="font-bold text-gray-800">${whatIfPrediction.totalCost.toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-gray-700">Inventory Turnover:</span>
                                    <span className="font-bold text-purple-600">{whatIfPrediction.turnover.toFixed(1)}x</span>
                                </div>
                            </div>
                        ) : whatIfQuantity && (parseInt(whatIfQuantity) < 0 || isNaN(parseInt(whatIfQuantity))) ? (
                            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                <p className="text-red-700 text-sm">⚠️ Please enter a valid quantity (0-10,000 units)</p>
                            </div>
                        ) : null}
                    </div>
                </div>

                {/* 7-Day Demand Forecast Chart */}
                <div className="mb-6">
                    <ComparisonChart
                        title="7-Day Demand Forecast"
                        type="line"
                        data={demandForecastData}
                        dataKey1="actual"
                        dataKey2="forecast"
                        label1="Historical Demand"
                        label2="Forecasted Demand"
                        yAxisLabel="Units/Day"
                    />
                </div>

                {/* Recommended Actions */}
                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Recommended Actions</h3>

                    {scenario === 'spike' && (
                        <div className="mb-4">
                            <div className="flex items-start bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded-r-lg mb-3">
                                <svg className="w-5 h-5 text-yellow-600 mr-3 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                                <div>
                                    <p className="font-semibold text-yellow-800">Action Required:</p>
                                    <ul className="list-disc list-inside mt-2 text-yellow-700 space-y-1">
                                        <li>Order {recommendation?.orderQuantity} units by tomorrow</li>
                                        <li>Increase safety stock to {Math.floor(currentStatus.safetyStock * 1.5)} units</li>
                                        <li>Monitor demand - spike expected Day +2</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    )}

                    {scenario === 'normal' && (
                        <div className="flex items-start bg-green-50 border-l-4 border-green-500 p-4 rounded-r-lg">
                            <svg className="w-5 h-5 text-green-600 mr-3 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <div>
                                <p className="font-semibold text-green-800">Current Performance:</p>
                                <ul className="list-disc list-inside mt-2 text-green-700 space-y-1">
                                    <li>Service level target: {(recommendation?.expectedServiceLevel! * 100).toFixed(1)}%</li>
                                    <li>Inventory coverage: {currentStatus.daysOfSupply} days</li>
                                    <li>Stockout risk: {recommendation?.stockoutRisk}</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {scenario === 'drop' && (
                        <div className="flex items-start bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg">
                            <svg className="w-5 h-5 text-blue-600 mr-3 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <div>
                                <p className="font-semibold text-blue-800">Optimization Opportunity:</p>
                                <ul className="list-disc list-inside mt-2 text-blue-700 space-y-1">
                                    <li>Reduce order to {recommendation?.orderQuantity} units to avoid overstock</li>
                                    <li>Projected savings: ${Math.floor((currentStatus.demandForecast * 14 - recommendation?.orderQuantity!) * 3.5).toLocaleString()}</li>
                                    <li>Monitor for demand recovery</li>
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default InventoryManager;
