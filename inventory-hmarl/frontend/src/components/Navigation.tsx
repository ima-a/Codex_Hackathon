import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
    const location = useLocation();

    const isActive = (path: string) => location.pathname === path;

    const navItems = [
        { path: '/', label: 'Overview' },
        { path: '/inventory-manager', label: 'Inventory Manager', highlight: true },
        { path: '/dashboard', label: 'Performance' },
        { path: '/agents', label: 'Agent Behavior' },
        { path: '/reconciliation', label: 'Reconciliation' },
    ];

    return (
        <nav className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    <div className="flex">
                        <div className="flex-shrink-0 flex items-center">
                            <h1 className="text-xl font-bold text-primary">
                                HMARL Supply Chain
                            </h1>
                        </div>
                        <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
                            {navItems.map((item) => (
                                <Link
                                    key={item.path}
                                    to={item.path}
                                    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${isActive(item.path)
                                            ? 'border-primary text-gray-900'
                                            : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                                        } ${item.highlight ? 'font-semibold' : ''}`}
                                >
                                    {item.label}
                                    {item.highlight && (
                                        <span className="ml-2 bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded-full">
                                            New
                                        </span>
                                    )}
                                </Link>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navigation;
