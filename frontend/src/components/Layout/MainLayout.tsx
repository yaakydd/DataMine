import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, UploadCloud, PieChart, Menu, X, Database, FolderOpen } from 'lucide-react';
import { cn } from '../../utils/cn';
import LearningSidebar from '../LearningPanel/LearningSidebar';

import WorkspaceSidebar from '../Workspace/WorkspaceSidebar';

export default function MainLayout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isLearningOpen, setIsLearningOpen] = useState(false);
  const [isWorkspaceOpen, setIsWorkspaceOpen] = useState(false);

  const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/upload', icon: UploadCloud, label: 'Upload Data' },
    { to: '/analysis', icon: PieChart, label: 'Analysis' },
  ];

  return (
    <div className="flex h-screen bg-background overflow-hidden relative">
      {/* Sidebar */}
      <aside
        className={cn(
          "absolute z-20 top-0 left-0 h-full w-64 bg-card border-r border-border transition-transform duration-300 ease-in-out md:relative md:translate-x-0",
          !isSidebarOpen && "-translate-x-full md:w-20"
        )}
      >
        <div className="flex items-center justify-between p-4 h-16 border-b border-border">
          <div className={cn("flex items-center gap-2 font-bold text-xl text-primary transition-opacity", !isSidebarOpen && "md:opacity-0 md:hidden")}>
             <Database className="w-6 h-6" />
             <span>FairData</span>
          </div>
           {/* Mobile close button */}
           <button onClick={() => setIsSidebarOpen(false)} className="md:hidden text-muted-foreground hover:text-foreground">
             <X className="w-6 h-6" />
           </button>
           {/* Desktop toggle button (collapsed state icon) */}
           {!isSidebarOpen && (
               <div className="hidden md:flex items-center justify-center w-full">
                   <Database className="w-6 h-6 text-primary" />
               </div>
           )}
        </div>

        <nav className="p-4 space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }: { isActive: boolean }) =>
                cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg transition-colors",
                  isActive
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    !isSidebarOpen && "justify-center"
                )
              }
              title={!isSidebarOpen ? item.label : undefined}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              <span className={cn("transition-opacity whitespace-nowrap", !isSidebarOpen && "md:hidden")}>
                {item.label}
              </span>
            </NavLink>
          ))}

          
          <button
            onClick={() => setIsWorkspaceOpen(!isWorkspaceOpen)}
            className={cn(
              "w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-muted-foreground hover:bg-muted hover:text-foreground",
              isWorkspaceOpen && "bg-primary/10 text-primary font-medium",
              !isSidebarOpen && "justify-center"
            )}
            title={!isSidebarOpen ? "Workspace" : undefined}
          >
            <FolderOpen className="w-5 h-5 flex-shrink-0" />
            <span className={cn("transition-opacity whitespace-nowrap", !isSidebarOpen && "md:hidden")}>
              Workspace
            </span>
          </button>
        </nav>
      </aside>

      {/* Workspace Sidebar (Explorer style) */}
      {isWorkspaceOpen && (
        <WorkspaceSidebar className="hidden md:flex border-r border-border" />
      )}

      {/* Main Content */}
      <main className={cn(
          "flex-1 flex flex-col min-w-0 bg-background overflow-hidden transition-all duration-300",
          isLearningOpen && "mr-80"
      )}>
        {/* Header (Mobile Toggle) */}
        <header className="flex md:hidden items-center p-4 border-b border-border bg-card">
            <button onClick={() => setIsSidebarOpen(true)} className="text-muted-foreground hover:text-foreground">
                <Menu className="w-6 h-6" />
            </button>
            <span className="ml-4 font-bold text-lg">FairData</span>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-auto p-4 md:p-8">
            <Outlet />
          <LearningSidebar 
        isOpen={isLearningOpen} 
        onToggle={() => setIsLearningOpen(!isLearningOpen)} 
      />
    </div>
      </main>
    </div>
  );
}
