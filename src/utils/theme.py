from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ThemeConfig:
    """主题配置类"""
    
    @staticmethod
    def get_theme_css() -> str:
        """获取主题CSS"""
        return """
            /* 主题变量 */
            :root {
                /* 主色调 */
                --primary-50: #eff6ff;
                --primary-100: #dbeafe;
                --primary-200: #bfdbfe;
                --primary-300: #93c5fd;
                --primary-400: #60a5fa;
                --primary-500: #3b82f6;
                --primary-600: #2563eb;
                --primary-700: #1d4ed8;
                --primary-800: #1e40af;
                --primary-900: #1e3a8a;
                
                /* 中性色 */
                --neutral-50: #f8fafc;
                --neutral-100: #f1f5f9;
                --neutral-200: #e2e8f0;
                --neutral-300: #cbd5e1;
                --neutral-400: #94a3b8;
                --neutral-500: #64748b;
                --neutral-600: #475569;
                --neutral-700: #334155;
                --neutral-800: #1e293b;
                --neutral-900: #0f172a;
                
                /* 功能色 */
                --success-500: #22c55e;
                --warning-500: #f59e0b;
                --error-500: #ef4444;
                --info-500: #3b82f6;
                
                /* 背景色 */
                --background-fill-primary: var(--neutral-50);
                --background-fill-secondary: var(--neutral-100);
                
                /* 文本色 */
                --body-text-color: var(--neutral-800);
                --heading-text-color: var(--neutral-900);
                --secondary-text-color: var(--neutral-600);
                
                /* 边框色 */
                --border-color-primary: var(--neutral-200);
                --border-color-secondary: var(--neutral-300);
                
                /* 动画变量 */
                --animation-duration: 0.3s;
                --animation-easing: cubic-bezier(0.4, 0, 0.2, 1);
                
                /* 阴影 */
                --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
                --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
                --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            }

            /* 深色模式 */
            [data-theme="dark"] {
                /* 主色调 */
                --primary-50: #172554;
                --primary-100: #1e3a8a;
                --primary-200: #1e40af;
                --primary-300: #1d4ed8;
                --primary-400: #2563eb;
                --primary-500: #3b82f6;
                --primary-600: #60a5fa;
                --primary-700: #93c5fd;
                --primary-800: #bfdbfe;
                --primary-900: #dbeafe;
                
                /* 中性色 */
                --neutral-50: #0f172a;
                --neutral-100: #1e293b;
                --neutral-200: #334155;
                --neutral-300: #475569;
                --neutral-400: #64748b;
                --neutral-500: #94a3b8;
                --neutral-600: #cbd5e1;
                --neutral-700: #e2e8f0;
                --neutral-800: #f1f5f9;
                --neutral-900: #f8fafc;
                
                /* 背景色 */
                --background-fill-primary: var(--neutral-100);
                --background-fill-secondary: var(--neutral-200);
                
                /* 文本色 */
                --body-text-color: var(--neutral-200);
                --heading-text-color: var(--neutral-100);
                --secondary-text-color: var(--neutral-400);
                
                /* 边框色 */
                --border-color-primary: var(--neutral-700);
                --border-color-secondary: var(--neutral-600);
                
                /* 阴影 */
                --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
                --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4);
                --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.5);
            }
        """
    
    @staticmethod
    def get_theme_toggle_js() -> str:
        """获取主题切换JS"""
        return """
            function toggleTheme() {
                const html = document.documentElement;
                const currentTheme = html.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                html.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                
                // 触发主题变更事件
                window.dispatchEvent(new Event('themechange'));
            }
            
            // 初始化主题
            function initTheme() {
                const savedTheme = localStorage.getItem('theme');
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                const theme = savedTheme || (prefersDark ? 'dark' : 'light');
                
                document.documentElement.setAttribute('data-theme', theme);
            }
            
            // 监听系统主题变化
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                if (!localStorage.getItem('theme')) {
                    document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
                }
            });
            
            // 页面加载时初始化主题
            document.addEventListener('DOMContentLoaded', initTheme);
        """