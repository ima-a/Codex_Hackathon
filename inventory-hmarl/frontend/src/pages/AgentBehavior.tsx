import { useState } from 'react';

type AgentType = 'store' | 'warehouse' | 'supplier';

const AgentBehavior = () => {
    const [activeTab, setActiveTab] = useState<AgentType>('store');

    const agents = {
        store: {
            title: 'Store Agent (PPO-Trained)',
            role: 'Decides when and how much to order from the warehouse to meet customer demand while minimizing costs.',
            learning: 'PPO with shared policy across all stores',
            observations: [
                'Current inventory level',
                'Demand forecast (7-day moving average)',
                'Reorder point (ROP)',
                'Safety stock level',
                'In-transit orders',
                'Backorders',
                'Day of week (seasonality)',
            ],
            actions: [
                'Order quantity (0-1000 units)',
                'Discrete action space with 4 options',
                'Adjusted based on inventory position',
            ],
            performance: {
                avgReward: '300.00',
                serviceLevel: '97.2%',
                trainingEpisodes: '334',
            },
        },
        warehouse: {
            title: 'Warehouse Agent (Rule-Based)',
            role: 'Distributes inventory to stores and places replenishment orders to the supplier.',
            learning: 'Classical (s,S) reorder point policy',
            observations: [
                'Current inventory by SKU',
                'Aggregate store demand',
                'In-transit from supplier',
                'Store backorders',
                'Inventory position',
            ],
            actions: [
                'Fulfill store orders (FIFO)',
                'Place supplier orders when inventory ≤ ROP',
                'Order up to target level (S)',
            ],
            performance: {
                avgReward: '150.00',
                fulfillmentRate: '98.5%',
                avgLeadTime: '2 days',
            },
        },
        supplier: {
            title: 'Supplier Agent (Rule-Based)',
            role: 'Produces goods and delivers to the warehouse after a fixed lead time.',
            learning: 'Deterministic FIFO fulfillment',
            observations: [
                'Pending orders queue',
                'Production capacity',
                'Lead time remaining',
            ],
            actions: [
                'Process orders in FIFO order',
                'Deliver after 7-day lead time',
                'Fulfill with 98% reliability',
            ],
            performance: {
                avgReward: '60.00',
                reliability: '98.0%',
                avgLeadTime: '7 days',
            },
        },
    };

    const currentAgent = agents[activeTab];

    return (
        <div className="min-h-screen bg-gray-50 py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-800">Agent Behavior</h1>
                    <p className="text-gray-600 mt-2">
                        Understanding how each agent makes decisions in the supply chain
                    </p>
                </div>

                {/* Tabs */}
                <div className="mb-8">
                    <div className="border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8">
                            {(['store', 'warehouse', 'supplier'] as AgentType[]).map((type) => (
                                <button
                                    key={type}
                                    onClick={() => setActiveTab(type)}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${activeTab === type
                                            ? 'border-blue-500 text-blue-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                        }`}
                                >
                                    {type.charAt(0).toUpperCase() + type.slice(1)} Agent
                                    {type === 'store' && (
                                        <span className="ml-2 bg-green-100 text-green-800 text-xs px-2 py-1 rounded">
                                            PPO
                                        </span>
                                    )}
                                </button>
                            ))}
                        </nav>
                    </div>
                </div>

                {/* Agent Details */}
                <div className="bg-white rounded-lg shadow-md p-8 border border-gray-200">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">
                        {currentAgent.title}
                    </h2>
                    <p className="text-lg text-gray-600 mb-8">{currentAgent.role}</p>

                    <div className="grid md:grid-cols-2 gap-8 mb-8">
                        {/* Observations */}
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                                <svg className="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                </svg>
                                Observations
                            </h3>
                            <ul className="space-y-2">
                                {currentAgent.observations.map((obs, idx) => (
                                    <li key={idx} className="flex items-start">
                                        <span className="text-blue-600 mr-2">•</span>
                                        <span className="text-gray-700">{obs}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Actions */}
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                                <svg className="w-5 h-5 mr-2 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                                Actions
                            </h3>
                            <ul className="space-y-2">
                                {currentAgent.actions.map((action, idx) => (
                                    <li key={idx} className="flex items-start">
                                        <span className="text-green-600 mr-2">•</span>
                                        <span className="text-gray-700">{action}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    {/* Learning Method */}
                    <div className="mb-8 p-4 bg-gray-50 rounded-lg">
                        <h3 className="text-lg font-semibold text-gray-800 mb-2">
                            Learning Method
                        </h3>
                        <p className="text-gray-700">{currentAgent.learning}</p>
                    </div>

                    {/* Performance Metrics */}
                    <div>
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">
                            Performance Metrics
                        </h3>
                        <div className="grid grid-cols-3 gap-4">
                            {Object.entries(currentAgent.performance).map(([key, value]) => (
                                <div key={key} className="bg-blue-50 p-4 rounded-lg">
                                    <p className="text-sm text-blue-600 font-medium capitalize">
                                        {key.replace(/([A-Z])/g, ' $1').trim()}
                                    </p>
                                    <p className="text-2xl font-bold text-blue-900 mt-1">{value}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Explanation Box */}
                <div className="mt-8 bg-yellow-50 border-l-4 border-yellow-500 p-6 rounded-r-lg">
                    <h3 className="text-lg font-semibold text-yellow-900 mb-2">
                        💡 Why Multi-Agent?
                    </h3>
                    <p className="text-yellow-800">
                        Each echelon in the supply chain (stores, warehouse, supplier) has different objectives
                        and constraints. Multi-agent RL allows each agent to learn independently while coordinating
                        through the shared environment. Store agents use PPO to learn complex demand patterns,
                        while warehouse and supplier use proven classical methods for stability.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default AgentBehavior;
