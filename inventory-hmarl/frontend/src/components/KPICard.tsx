interface KPICardProps {
    title: string;
    baseline: number;
    ppo: number;
    unit: string;
    improvement: 'higher' | 'lower';
    format?: 'number' | 'percentage' | 'currency';
}

const KPICard = ({ title, baseline, ppo, unit, improvement, format = 'number' }: KPICardProps) => {
    const formatValue = (value: number) => {
        if (format === 'percentage') {
            return `${(value * 100).toFixed(1)}%`;
        } else if (format === 'currency') {
            return `$${(value / 1000).toFixed(1)}K`;
        }
        return value.toFixed(0);
    };

    const delta = ppo - baseline;
    const percentChange = ((delta / baseline) * 100).toFixed(1);

    const isImprovement = improvement === 'higher' ? delta > 0 : delta < 0;
    const changeColor = isImprovement ? 'text-success' : 'text-danger';
    const arrow = improvement === 'higher'
        ? (delta > 0 ? '↑' : '↓')
        : (delta < 0 ? '↓' : '↑');

    return (
        <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow">
            <h3 className="text-sm font-medium text-gray-600 mb-2">{title}</h3>

            <div className="grid grid-cols-2 gap-4 mb-3">
                <div>
                    <p className="text-xs text-gray-500">Baseline</p>
                    <p className="text-2xl font-bold text-gray-700">
                        {formatValue(baseline)}
                    </p>
                </div>
                <div>
                    <p className="text-xs text-gray-500">PPO</p>
                    <p className="text-2xl font-bold text-primary">
                        {formatValue(ppo)}
                    </p>
                </div>
            </div>

            <div className={`flex items-center text-sm font-semibold ${changeColor}`}>
                <span className="text-lg mr-1">{arrow}</span>
                <span>{Math.abs(parseFloat(percentChange))}% {isImprovement ? 'improvement' : 'worse'}</span>
            </div>
        </div>
    );
};

export default KPICard;
