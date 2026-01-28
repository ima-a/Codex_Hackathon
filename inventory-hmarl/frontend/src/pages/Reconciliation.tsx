const Reconciliation = () => {
    const sampleReconciliation = [
        {
            metric: 'Demand',
            planned: 50,
            actual: 75,
            delta: '+25',
            deltaPercent: '+50%',
            reasonCode: 'Demand Spike',
            severity: 'critical',
        },
        {
            metric: 'Service Level',
            planned: 95,
            actual: 93,
            delta: '-2',
            deltaPercent: '-2.1%',
            reasonCode: 'Stockout',
            severity: 'warning',
        },
        {
            metric: 'Holding Cost',
            planned: 50,
            actual: 45,
            delta: '-5',
            deltaPercent: '-10%',
            reasonCode: 'Good Performance',
            severity: 'good',
        },
        {
            metric: 'Inventory',
            planned: 200,
            actual: 185,
            delta: '-15',
            deltaPercent: '-7.5%',
            reasonCode: 'On Target',
            severity: 'info',
        },
    ];

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical':
                return 'bg-red-100 text-red-800';
            case 'warning':
                return 'bg-yellow-100 text-yellow-800';
            case 'good':
                return 'bg-green-100 text-green-800';
            default:
                return 'bg-blue-100 text-blue-800';
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-800">Reconciliation</h1>
                    <p className="text-gray-600 mt-2">
                        Comparing planned vs actual outcomes to understand system performance
                    </p>
                </div>

                {/* Explanation */}
                <div className="bg-white rounded-lg shadow-md p-8 border border-gray-200 mb-8">
                    <h2 className="text-2xl font-bold text-gray-800 mb-4">
                        What is Reconciliation?
                    </h2>
                    <div className="prose max-w-none text-gray-700">
                        <p className="text-lg mb-4">
                            After each day of simulation, we compare what the AI agents <strong>planned</strong> to
                            happen with what <strong>actually</strong> happened. This tells us:
                        </p>
                        <ul className="list-disc list-inside space-y-2 mb-6">
                            <li>Why the system performs well or poorly</li>
                            <li>Which decisions led to good outcomes</li>
                            <li>What external factors (demand spikes, delays) affected performance</li>
                            <li>How to improve the AI agents' learning</li>
                        </ul>

                        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg">
                            <h3 className="font-semibold text-blue-900 mb-2">
                                🔗 Bridging Business Metrics and AI Rewards
                            </h3>
                            <p className="text-blue-800">
                                Reconciliation converts business outcomes (service level, costs) into reward signals
                                for reinforcement learning. High service + low costs = positive reward.
                                Stockouts + high inventory = negative reward.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Metrics Table */}
                <div className="bg-white rounded-lg shadow-md border border-gray-200 mb-8">
                    <div className="px-6 py-4 border-b border-gray-200">
                        <h3 className="text-lg font-semibold text-gray-800">
                            Sample Reconciliation Report
                        </h3>
                        <p className="text-sm text-gray-600 mt-1">
                            Example from a single day of simulation
                        </p>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Metric
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Planned
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Actual
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Delta
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Reason Code
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {sampleReconciliation.map((row, idx) => (
                                    <tr key={idx} className="hover:bg-gray-50">
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                            {row.metric}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                                            {row.planned}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                                            {row.actual}
                                        </td>
                                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${row.delta.startsWith('+') ? 'text-red-600' :
                                                row.delta.startsWith('-') && row.metric.includes('Cost') ? 'text-green-600' :
                                                    'text-gray-600'
                                            }`}>
                                            {row.delta} <span className="text-xs">({row.deltaPercent})</span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            <span className={`px-3 py-1 rounded-full text-xs font-medium ${getSeverityColor(row.severity)}`}>
                                                {row.reasonCode}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Reason Codes */}
                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 mb-8">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                        Common Deviation Reason Codes
                    </h3>
                    <div className="grid md:grid-cols-2 gap-4">
                        <div>
                            <h4 className="font-medium text-red-800 mb-2">🔴 Critical</h4>
                            <ul className="text-sm text-gray-700 space-y-1">
                                <li>• Demand Spike - Unexpected surge in demand</li>
                                <li>• Upstream Shortage - Supplier couldn't fulfill</li>
                                <li>• Insufficient Safety Stock - Buffer too low</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-medium text-yellow-800 mb-2">🟡 Warning</h4>
                            <ul className="text-sm text-gray-700 space-y-1">
                                <li>• Forecast Error - Prediction inaccurate</li>
                                <li>• Policy Under-Ordering - Ordered too little</li>
                                <li>• Order Not Fulfilled - Logistics issue</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-medium text-blue-800 mb-2">ℹ️ Info</h4>
                            <ul className="text-sm text-gray-700 space-y-1">
                                <li>• Demand Drop - Lower than expected demand</li>
                                <li>• Policy Over-Ordering - Ordered too much</li>
                                <li>• Execution Delay - Timing issue</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-medium text-green-800 mb-2">🟢 Good</h4>
                            <ul className="text-sm text-gray-700 space-y-1">
                                <li>• Perfect Execution - Plan matched reality</li>
                                <li>• On Target - Within acceptable variance</li>
                                <li>• Good Performance - Optimal outcome</li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Key Insight */}
                <div className="bg-green-50 border-l-4 border-green-500 p-6 rounded-r-lg">
                    <h3 className="text-lg font-semibold text-green-900 mb-2">
                        🎯 Why This Matters
                    </h3>
                    <p className="text-green-800">
                        Reconciliation makes AI explainable to business stakeholders. Instead of a black box,
                        decision-makers can see exactly <em>why</em> the AI made specific choices and
                        <em>what</em> the outcomes were. This builds trust and enables continuous improvement.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Reconciliation;
