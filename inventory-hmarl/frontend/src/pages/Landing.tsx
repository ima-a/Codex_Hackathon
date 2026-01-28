import { Link } from 'react-router-dom';

const Landing = () => {
    return (
        <div className="min-h-screen">
            {/* Hero Section */}
            <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
                    <div className="text-center">
                        <h1 className="text-5xl font-extrabold mb-4">
                            AI-Driven Digital Twin for Retail Inventory Optimization
                        </h1>
                        <p className="text-xl mb-8 text-blue-100">
                            Hierarchical Multi-Agent Reinforcement Learning outperforms baseline policies by 30%
                        </p>
                        <Link
                            to="/dashboard"
                            className="inline-block bg-white text-blue-600 px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-50 transition-colors shadow-lg"
                        >
                            View Results Dashboard →
                        </Link>
                    </div>
                </div>
            </div>

            {/* What is This Section */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
                <h2 className="text-3xl font-bold text-center text-gray-800 mb-12">
                    How It Works
                </h2>
                <div className="grid md:grid-cols-3 gap-8">
                    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                            <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                            </svg>
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Digital Twin Simulation</h3>
                        <p className="text-gray-600">
                            Realistic supply chain model with 3 stores, 1 warehouse, and 1 supplier.
                            Simulates stochastic demand, lead times, and inventory dynamics.
                        </p>
                    </div>

                    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                        <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center mb-4">
                            <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                            </svg>
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Baseline Policies</h3>
                        <p className="text-gray-600">
                            Classical inventory control using reorder point (ROP) and safety stock formulas.
                            Provides benchmark performance.
                        </p>
                    </div>

                    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                        <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                            <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                        </div>
                        <h3 className="text-xl font-semibold mb-2">PPO-Trained Agents</h3>
                        <p className="text-gray-600">
                            Multi-agent reinforcement learning with Proximal Policy Optimization.
                            Learns optimal inventory decisions from experience.
                        </p>
                    </div>
                </div>
            </div>

            {/* Architecture Section */}
            <div className="bg-gray-100 py-16">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">
                        System Architecture
                    </h2>
                    <div className="bg-white p-8 rounded-lg shadow-md">
                        <div className="text-center">
                            <div className="mb-8">
                                <div className="inline-block bg-blue-50 border-2 border-blue-500 rounded-lg p-4 mb-4">
                                    <p className="font-semibold text-blue-900">Supplier Agent (Rule-based)</p>
                                    <p className="text-sm text-blue-700">Production & Fulfillment</p>
                                </div>
                            </div>

                            <div className="mb-8">
                                <svg className="w-6 h-6 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                                </svg>
                            </div>

                            <div className="mb-8">
                                <div className="inline-block bg-gray-50 border-2 border-gray-500 rounded-lg p-4 mb-4">
                                    <p className="font-semibold text-gray-900">Warehouse Agent (Rule-based)</p>
                                    <p className="text-sm text-gray-700">Distribution & Replenishment</p>
                                </div>
                            </div>

                            <div className="mb-8">
                                <svg className="w-6 h-6 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                                </svg>
                            </div>

                            <div className="flex justify-center gap-4">
                                <div className="bg-green-50 border-2 border-green-500 rounded-lg p-4">
                                    <p className="font-semibold text-green-900">Store 1 (PPO)</p>
                                    <p className="text-sm text-green-700">Retail Agent</p>
                                </div>
                                <div className="bg-green-50 border-2 border-green-500 rounded-lg p-4">
                                    <p className="font-semibold text-green-900">Store 2 (PPO)</p>
                                    <p className="text-sm text-green-700">Retail Agent</p>
                                </div>
                                <div className="bg-green-50 border-2 border-green-500 rounded-lg p-4">
                                    <p className="font-semibold text-green-900">Store 3 (PPO)</p>
                                    <p className="text-sm text-green-700">Retail Agent</p>
                                </div>
                            </div>

                            <p className="mt-8 text-gray-600">
                                <span className="font-semibold text-green-600">Green</span>: PPO-trained agents |
                                <span className="font-semibold text-gray-600"> Gray/Blue</span>: Rule-based policies
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* CTA Section */}
            <div className="bg-white py-16">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <h2 className="text-3xl font-bold text-gray-800 mb-4">
                        Ready to See the Results?
                    </h2>
                    <p className="text-xl text-gray-600 mb-8">
                        Explore how AI agents achieve 30% better performance than baseline policies
                    </p>
                    <Link
                        to="/dashboard"
                        className="inline-block bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors shadow-lg"
                    >
                        View Results Dashboard →
                    </Link>
                </div>
            </div>
        </div>
    );
};

export default Landing;
