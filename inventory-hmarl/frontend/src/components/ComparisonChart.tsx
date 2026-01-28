import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    LineChart,
    Line,
} from 'recharts';

interface ComparisonChartProps {
    title: string;
    type: 'bar' | 'line';
    data: any[];
    dataKey1: string;
    dataKey2?: string;
    label1: string;
    label2?: string;
    yAxisLabel?: string;
}

const ComparisonChart = ({
    title,
    type,
    data,
    dataKey1,
    dataKey2,
    label1,
    label2,
    yAxisLabel,
}: ComparisonChartProps) => {
    return (
        <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
            <ResponsiveContainer width="100%" height={300}>
                {type === 'bar' ? (
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey={dataKey1} fill="#9CA3AF" name={label1} />
                        {dataKey2 && <Bar dataKey={dataKey2} fill="#3B82F6" name={label2} />}
                    </BarChart>
                ) : (
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey={dataKey1} stroke="#3B82F6" strokeWidth={2} name={label1} />
                        {dataKey2 && <Line type="monotone" dataKey={dataKey2} stroke="#10B981" strokeWidth={2} name={label2} />}
                    </LineChart>
                )}
            </ResponsiveContainer>
        </div>
    );
};

export default ComparisonChart;
