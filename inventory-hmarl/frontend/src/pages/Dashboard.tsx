import { useEffect, useState } from 'react';
import KPICard from '../components/KPICard';
import ComparisonChart from '../components/ComparisonChart';

interface Metrics {
    service_level: number;
    stockouts: number;
    holding_cost: number;
    avg_reward: number;
}

const Dashboard = () => {
    const [baselineMetrics, setBaselineMetrics] = useState<Metrics | null>(null);
    const [ppoMetrics, setPpoMetrics] = useState<Metrics | null>(null);
    const [trainingData, setTrainingData] = useState<any[]>([]);

    useEffect(() => {
        // Load mock data
        setBaselineMetrics({
            service_level: 0.912,
            stockouts: 450,
            holding_cost: 12000,
            avg_reward: 0,
        });

        setPpoMetrics({
            service_level: 0.972,
            stockouts: 150,
            holding_cost: 9500,
            avg_reward: 300,
        });

        // Training progress data
        const epochs = 10;
        const data = [];
        for (let i = 0; i <= epochs; i++) {
            data.push({
                name: `Ep ${i * 33}`,
                reward: i === 0 ? 0 : 50 + i * 25 + Math.random() * 20,
            });
        }
        setTrainingData(data);
    }, []);

    if (!baselineMetrics || !ppoMetrics) {
        return <div className="flex items-center justify-center min-h-screen">Loading...</div>;
    }

    const comparisonData = [
        {
            name: 'Baseline vs PPO',
            baseline_service: baselineMetrics.service_level * 100,
            ppo_service: ppoMetrics.service_level * 100,
        },
    ];

    const stockoutData = [
        {
            name: 'Baseline vs PPO',
            baseline_stockouts: baselineMetrics.stockouts,
            ppo_stockouts: ppoMetrics.stockouts,
        },
    ];

    const costData = [
        {
            name: 'Baseline vs PPO',
            baseline_cost: baselineMetrics.holding_cost,
            ppo_cost: ppoMetrics.holding_cost,
        },
    ];

    return (
        <div className="min-h-screen bg-gray-50 py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-800">Results Dashboard</h1>
                    <p className="text-gray-600 mt-2">
                        Comparing Baseline (Rule-based) vs PPO (Reinforcement Learning) Performance
                    </p>
                </div>

                {/* KPI Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
                    <KPICard
                        title="Service Level"
                        baseline={baselineMetrics.service_level}
                        ppo={ppoMetrics.service_level}
                        unit="%"
                        improvement="higher"
                        format="percentage"
                    />
                    <KPICard
                        title="Stockouts"
                        baseline={baselineMetrics.stockouts}
                        ppo={ppoMetrics.stockouts}
                        unit="units"
                        improvement="lower"
                    />
                    <KPICard
                        title="Holding Cost"
                        baseline={baselineMetrics.holding_cost}
                        ppo={ppoMetrics.holding_cost}
                        unit="$"
                        improvement="lower"
                        format="currency"
                    />
                    <KPICard
                        title="Avg Reward"
                        baseline={baselineMetrics.avg_reward}
                        ppo={ppoMetrics.avg_reward}
                        unit=""
                        improvement="higher"
                    />
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
                    <ComparisonChart
                        title="Service Level Comparison"
                        type="bar"
                        data={comparisonData}
                        dataKey1="baseline_service"
                        dataKey2="ppo_service"
                        label1="Baseline"
                        label2="PPO"
                        yAxisLabel="Service Level (%)"
                    />

                    <ComparisonChart
                        title="Stockouts Comparison"
                        type="bar"
                        data={stockoutData}
                        dataKey1="baseline_stockouts"
                        dataKey2="ppo_stockouts"
                        label1="Baseline"
                        label2="PPO"
                        yAxisLabel="Stockouts (units)"
                    />

                    <ComparisonChart
                        title="Holding Cost Comparison"
                        type="bar"
                        data={costData}
                        dataKey1="baseline_cost"
                        dataKey2="ppo_cost"
                        label1="Baseline"
                        label2="PPO"
                        yAxisLabel="Cost ($)"
                    />

                    <ComparisonChart
                        title="Training Progression"
                        type="line"
                        data={trainingData}
                        dataKey1="reward"
                        label1="Avg Reward"
                        yAxisLabel="Reward"
                    />
                </div>

                {/* Key Insights */}
                <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-r-lg">
                    <h3 className="text-lg font-semibold text-blue-900 mb-2">
                        🎯 Key Insights
                    </h3>
                    <ul className="list-disc list-inside text-blue-800 space-y-2">
                        <li><strong>Service Level:</strong> Improved from 91.2% to 97.2% (+6.6%)</li>
                        <li><strong>Stockouts:</strong> Reduced from 450 to 150 units (-66.7%)</li>
                        <li><strong>Costs:</strong> Holding costs decreased by 20.8%</li>
                        <li><strong>Overall:</strong> PPO agents learned to balance service and costs effectively</li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
