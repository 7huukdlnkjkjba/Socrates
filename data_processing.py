import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from statsmodels.tsa.stattools import adfuller
import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.data = None
        self.time_column = None
        self.target_column = None
        self._financial_sources = {
            'gold9999': {
                'name': '黄金9999',
                'url': 'https://finance.sina.com.cn/futures/quotes/AUTD.shtml',
                'historical_url': 'https://finance.sina.com.cn/futures/quotes_history/AUTD.shtml'
            },
            'gold_london': {
                'name': '伦敦金现',
                'url': 'https://finance.sina.com.cn/futures/quotes/XAUUSD.shtml',
                'historical_url': 'https://finance.sina.com.cn/futures/quotes_history/XAUUSD.shtml'
            },
            'usdcny': {
                'name': '美元兑人民币汇率',
                'url': 'https://finance.sina.com.cn/forex/quotes/USDCNY.shtml',
                'historical_url': 'https://finance.sina.com.cn/forex/quotes_history/USDCNY.shtml'
            },
            'boshi_gold_c': {
                'name': '博时黄金C',
                'url': 'https://finance.sina.com.cn/fund/quotes/002611.shtml',
                'historical_url': 'https://finance.sina.com.cn/fund/quotes_history/002611.shtml'
            }
        }
    
    def load_data(self, source, time_column, target_column, source_type='file', file_format='csv', crawl_mode='basic', **kwargs):
        """加载时间序列数据
        
        参数:
        source: 文件路径或URL
        time_column: 时间列名称
        target_column: 目标列名称
        source_type: 数据源类型 ('file' 或 'web')
        file_format: 文件格式 (仅当 source_type='file' 时有效)
        crawl_mode: 爬取模式 ('basic' 或 'web_driver')
        **kwargs: 额外参数
        """
        if source_type == 'file':
            # 从本地文件加载
            if file_format == 'csv':
                self.data = pd.read_csv(source, **kwargs)
            elif file_format == 'excel':
                self.data = pd.read_excel(source, **kwargs)
            elif file_format == 'json':
                self.data = pd.read_json(source, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        elif source_type == 'web':
            # 从网络爬取数据
            if crawl_mode == 'basic':
                self.data = self._crawl_data(source, **kwargs)
            elif crawl_mode == 'web_driver':
                self.data = self._crawl_with_web_driver(source, **kwargs)
            else:
                raise ValueError(f"Unsupported crawl mode: {crawl_mode}")
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        self.time_column = time_column
        self.target_column = target_column
        
        # 转换时间列为 datetime 类型
        self.data[self.time_column] = pd.to_datetime(self.data[self.time_column])
        
        # 设置时间列为索引
        self.data.set_index(self.time_column, inplace=True)
        
        return self.data
    
    def _crawl_data(self, url, retries=3, delay=2, **kwargs):
        """从指定URL爬取时间序列数据（基础爬取）"""
        # 通用爬虫逻辑，带重试机制
        for attempt in range(retries):
            try:
                response = requests.get(url, **kwargs)
                response.raise_for_status()  # 确保请求成功
                
                # 检查内容类型和URL扩展
                content_type = response.headers.get('Content-Type', '')
                url_lower = url.lower()
                
                # 优先根据URL扩展判断
                if url_lower.endswith('.csv') or 'csv' in content_type:
                    # CSV格式数据
                    import io
                    data = pd.read_csv(io.StringIO(response.text), **kwargs)
                elif url_lower.endswith('.json') or 'json' in content_type:
                    # JSON格式数据
                    data = pd.read_json(response.content, **kwargs)
                else:
                    # 尝试解析HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 查找表格
                    table = soup.find('table')
                    if table:
                        data = pd.read_html(str(table))[0]
                    else:
                        # 尝试直接使用pd.read_csv解析（处理GitHub Raw等情况）
                        try:
                            import io
                            data = pd.read_csv(io.StringIO(response.text), **kwargs)
                        except:
                            # 尝试使用正则表达式提取数据（简单情况）
                            import re
                            # 这是一个示例，需要根据具体网站调整
                            pattern = re.compile(r'\[("[\d-]+"\s*,\s*\d+)\]')
                            matches = pattern.findall(response.text)
                            if matches:
                                rows = []
                                for match in matches:
                                    date, value = match.split(',')
                                    rows.append({'date': date.strip('"'), 'value': float(value.strip())})
                                data = pd.DataFrame(rows)
                            else:
                                raise ValueError(f"无法从URL提取数据: {url}")
                
                return data
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay + random.uniform(0, 1))
                    delay *= 1.5  # 指数退避
                else:
                    raise ValueError(f"爬取数据失败，已尝试{retries}次: {str(e)}")
    
    def _crawl_with_web_driver(self, url, retries=3, delay=5, wait_time=10, table_selector=None, browser_type='chrome', headless=True, **kwargs):
        """使用浏览器驱动爬取动态生成的时间序列数据（BotBrowser风格）
        
        参数:
        url: 要爬取的URL
        retries: 重试次数
        delay: 重试延迟（秒）
        wait_time: 页面加载等待时间（秒）
        table_selector: 表格CSS选择器
        browser_type: 浏览器类型 ('chrome' 或 'firefox')
        headless: 是否使用无头模式
        **kwargs: 额外参数，包括:
            - click_selector: 需要点击的元素选择器
            - fill_form: 表单填写配置，格式为 {selector: value}
            - scroll_down: 是否向下滚动页面
            - scroll_count: 滚动次数
        """
        logger.info(f"开始使用{browser_type}浏览器爬取: {url}")
        
        for attempt in range(retries):
            try:
                # 初始化浏览器
                driver = self._init_browser(browser_type, headless, **kwargs)
                
                try:
                    # 打开URL
                    driver.get(url)
                    logger.info(f"成功加载页面: {url}")
                    
                    # 等待页面加载
                    WebDriverWait(driver, wait_time).until(
                        EC.presence_of_element_located((By.TAG_NAME, 'body'))
                    )
                    
                    # 执行额外操作（如点击元素）
                    if 'click_selector' in kwargs:
                        self._click_element(driver, kwargs['click_selector'], wait_time)
                        
                    # 填写表单
                    if 'fill_form' in kwargs:
                        self._fill_form(driver, kwargs['fill_form'], wait_time)
                    
                    # 滚动页面
                    if kwargs.get('scroll_down', False):
                        self._scroll_page(driver, kwargs.get('scroll_count', 3))
                    
                    # 如果指定了表格选择器，等待表格加载
                    if table_selector:
                        WebDriverWait(driver, wait_time).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
                        )
                    else:
                        # 否则等待一段时间让页面动态内容加载
                        time.sleep(3)
                    
                    # 获取页面HTML
                    page_source = driver.page_source
                    
                    # 尝试解析表格
                    data = self._extract_table_data(page_source, table_selector)
                    if data is not None:
                        logger.info(f"成功提取表格数据，形状: {data.shape}")
                        return data
                    
                    # 尝试提取JSON数据
                    data = self._extract_json_data(page_source)
                    if data is not None:
                        logger.info(f"成功提取JSON数据，形状: {data.shape}")
                        return data
                    
                    # 尝试提取自定义选择器数据
                    if 'data_selector' in kwargs:
                        data = self._extract_custom_data(driver, kwargs['data_selector'])
                        if data is not None:
                            logger.info(f"成功提取自定义选择器数据，形状: {data.shape}")
                            return data
                    
                    raise ValueError(f"无法从动态页面提取数据: {url}")
                    
                finally:
                    # 关闭浏览器
                    driver.quit()
                    logger.info("浏览器已关闭")
                    
            except Exception as e:
                logger.warning(f"爬取尝试{attempt+1}/{retries}失败: {str(e)}")
                if attempt < retries - 1:
                    sleep_time = delay + random.uniform(0, 2)
                    logger.info(f"等待{sleep_time:.1f}秒后重试...")
                    time.sleep(sleep_time)
                    delay *= 1.5  # 指数退避
                else:
                    logger.error(f"所有爬取尝试都失败了: {str(e)}")
                    raise ValueError(f"浏览器爬取数据失败，已尝试{retries}次: {str(e)}")
    
    def _init_browser(self, browser_type='chrome', headless=True, **kwargs):
        """初始化浏览器驱动（BotBrowser风格）"""
        if browser_type == 'chrome':
            options = webdriver.ChromeOptions()
            if headless:
                options.add_argument('--headless')
            
            # BotBrowser风格的浏览器指纹伪装选项
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')  # 禁用自动化检测
            options.add_argument('--disable-features=WebRtcHideLocalIpsWithMdns')  # 控制WebRTC本地IP暴露
            options.add_argument('--disable-audio-output')  # 禁用音频输出
            options.add_argument('--disable-crash-reporter')
            options.add_argument('--disable-crashpad-for-testing')
            options.add_argument('--disable-gpu-watchdog')
            
            # 添加代理（如果指定）
            if 'proxy' in kwargs:
                options.add_argument(f'--proxy-server={kwargs["proxy"]}')
            
            # 设置用户代理
            user_agent = kwargs.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.3595.65 Safari/537.36')
            options.add_argument(f'--user-agent={user_agent}')
            
            # 模拟真实窗口大小
            window_size = kwargs.get('window_size', '1280,720')
            options.add_argument(f'--window-size={window_size}')
            
            # Chrome特定的指纹伪装
            options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_experimental_option('prefs', {
                'intl.accept_languages': kwargs.get('languages', 'en-US,en'),
                'profile.default_content_setting_values.webrtc': 2,  # 禁用WebRTC
                'profile.default_content_setting_values.geolocation': 2,  # 禁用地理位置
            })
            
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
            
            # 进一步隐藏自动化特征 - BotBrowser风格的高级反检测
            driver.execute_script("""
                // 隐藏webdriver属性
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                
                // 模拟真实浏览器语言设置
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en', 'fr-FR']});
                
                // 模拟真实插件数量
                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        const plugins = [];
                        const pluginNames = ['Chrome PDF Plugin', 'Chrome PDF Viewer', 'Native Client', 'Widevine Content Decryption Module', 'Shockwave Flash'];
                        const pluginDescs = ['Portable Document Format', 'Portable Document Format', 'Native Client', 'Widevine Content Decryption Module', 'Shockwave Flash'];
                        const numPlugins = Math.floor(Math.random() * 3) + 2;
                        for (let i = 0; i < numPlugins; i++) {
                            plugins.push({
                                name: pluginNames[i % pluginNames.length],
                                description: pluginDescs[i % pluginDescs.length],
                                filename: pluginNames[i % pluginNames.length].replace(/\\s+/g, '') + '.dll'
                            });
                        }
                        return plugins;
                    }
                });
                
                // 模拟真实mime类型
                Object.defineProperty(navigator, 'mimeTypes', {
                    get: () => {
                        const mimeTypes = [];
                        const mimes = [
                            {type: 'text/html', suffixes: 'html,htm'},
                            {type: 'application/json', suffixes: 'json'},
                            {type: 'image/jpeg', suffixes: 'jpeg,jpg'},
                            {type: 'image/png', suffixes: 'png'},
                            {type: 'video/mp4', suffixes: 'mp4'},
                            {type: 'application/pdf', suffixes: 'pdf'},
                            {type: 'application/x-shockwave-flash', suffixes: 'swf'}
                        ];
                        for (const mime of mimes) {
                            mimeTypes.push(mime);
                        }
                        return mimeTypes;
                    }
                });
                
                // 修改window.navigator属性
                Object.defineProperty(navigator, 'userAgentData', {
                    get: () => {
                        return {
                            brands: [
                                {brand: 'Google Chrome', version: '142'},
                                {brand: 'Not A;Brand', version: '99'},
                                {brand: 'Chromium', version: '142'}
                            ],
                            platform: 'Windows',
                            platformVersion: '10.0.0',
                            architecture: 'x86',
                            model: '',
                            mobile: false,
                            wow64: false,
                            fullVersionList: [
                                {brand: 'Google Chrome', version: '142.0.3595.65'},
                                {brand: 'Not A;Brand', version: '99.0.0.0'},
                                {brand: 'Chromium', version: '142.0.3595.65'}
                            ]
                        };
                    }
                });
                
                // 模拟真实的screen属性
                Object.defineProperty(screen, 'width', {get: () => 1920});
                Object.defineProperty(screen, 'height', {get: () => 1080});
                Object.defineProperty(screen, 'colorDepth', {get: () => 24});
                Object.defineProperty(screen, 'pixelDepth', {get: () => 24});
                Object.defineProperty(screen, 'availWidth', {get: () => 1920});
                Object.defineProperty(screen, 'availHeight', {get: () => 1040});
                Object.defineProperty(screen, 'availLeft', {get: () => 0});
                Object.defineProperty(screen, 'availTop', {get: () => 0});
                
                // 模拟真实的window属性
                Object.defineProperty(window, 'innerWidth', {get: () => 1280});
                Object.defineProperty(window, 'innerHeight', {get: () => 720});
                Object.defineProperty(window, 'outerWidth', {get: () => 1280});
                Object.defineProperty(window, 'outerHeight', {get: () => 760});
                Object.defineProperty(window, 'screenX', {get: () => 100});
                Object.defineProperty(window, 'screenY', {get: () => 50});
                Object.defineProperty(window, 'devicePixelRatio', {get: () => 1});
                
                // 模拟真实的document属性
                Object.defineProperty(document, 'visibilityState', {get: () => 'visible'});
                Object.defineProperty(document, 'hidden', {get: () => false});
                Object.defineProperty(document, 'hasFocus', {value: () => true});
                
                // 移除playwright或其他框架的痕迹
                if (window.__playwright__) {
                    delete window.__playwright__;
                }
                if (window.__pwInitScripts__) {
                    delete window.__pwInitScripts__;
                }
                if (window.cdp) {
                    delete window.cdp;
                }
                if (window.__selenium_script__) {
                    delete window.__selenium_script__;
                }
                if (window.callPhantom) {
                    delete window.callPhantom;
                }
                if (window._phantom) {
                    delete window._phantom;
                }
                
                // 模拟真实的performance属性
                Object.defineProperty(performance, 'now', {
                    value: () => Date.now() + Math.random() * 100
                });
                

                
                // Canvas噪声注入 - BotBrowser风格（增强版）
                if (typeof HTMLCanvasElement !== 'undefined') {
                    const originalGetContext = HTMLCanvasElement.prototype.getContext;
                    HTMLCanvasElement.prototype.getContext = function(contextType, contextAttributes) {
                        const ctx = originalGetContext.call(this, contextType, contextAttributes);
                        if (ctx && contextType === '2d') {
                            // 增强的随机偏移
                            const originalFillRect = ctx.fillRect;
                            ctx.fillRect = function(x, y, width, height) {
                                const xOffset = Math.random() * 0.2 - 0.1;
                                const yOffset = Math.random() * 0.2 - 0.1;
                                originalFillRect.call(this, x + xOffset, y + yOffset, width, height);
                            };
                            
                            const originalStrokeRect = ctx.strokeRect;
                            ctx.strokeRect = function(x, y, width, height) {
                                const xOffset = Math.random() * 0.2 - 0.1;
                                const yOffset = Math.random() * 0.2 - 0.1;
                                originalStrokeRect.call(this, x + xOffset, y + yOffset, width, height);
                            };
                            
                            // 增强的像素噪声
                            const originalGetImageData = ctx.getImageData;
                            ctx.getImageData = function(sx, sy, sw, sh) {
                                const imageData = originalGetImageData.call(this, sx, sy, sw, sh);
                                
                                // 随机选择噪声模式
                                const noiseMode = Math.floor(Math.random() * 3);
                                
                                // 添加更真实的噪声到像素数据
                                for (let i = 0; i < imageData.data.length; i += 4) {
                                    let rNoise = 0, gNoise = 0, bNoise = 0;
                                    
                                    if (noiseMode === 0) {
                                        // 随机像素偏移
                                        rNoise = Math.floor(Math.random() * 3) - 1;
                                        gNoise = Math.floor(Math.random() * 3) - 1;
                                        bNoise = Math.floor(Math.random() * 3) - 1;
                                    } else if (noiseMode === 1) {
                                        // 轻微的颜色偏移
                                        const offset = Math.floor(Math.random() * 2) - 1;
                                        rNoise = offset;
                                        gNoise = offset;
                                        bNoise = offset;
                                    } else {
                                        // 区域噪声
                                        if (Math.random() < 0.1) {
                                            rNoise = Math.floor(Math.random() * 5) - 2;
                                            gNoise = Math.floor(Math.random() * 5) - 2;
                                            bNoise = Math.floor(Math.random() * 5) - 2;
                                        }
                                    }
                                    
                                    imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + rNoise));
                                    imageData.data[i + 1] = Math.max(0, Math.min(255, imageData.data[i + 1] + gNoise));
                                    imageData.data[i + 2] = Math.max(0, Math.min(255, imageData.data[i + 2] + bNoise));
                                }
                                return imageData;
                            };
                        }
                        return ctx;
                    };
                }
                
                // WebGL噪声注入 - BotBrowser风格
                if (typeof WebGLRenderingContext !== 'undefined') {
                    const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(pname) {
                        const result = originalGetParameter.call(this, pname);
                        if (pname === WebGLRenderingContext.VENDOR) {
                            return 'Google Inc.';
                        }
                        if (pname === WebGLRenderingContext.RENDERER) {
                            return 'ANGLE (NVIDIA GeForce GTX 1060 6GB Direct3D11 vs_5_0 ps_5_0)';
                        }
                        return result;
                    };
                    
                    const originalReadPixels = WebGLRenderingContext.prototype.readPixels;
                    WebGLRenderingContext.prototype.readPixels = function(x, y, width, height, format, type, pixels) {
                        const result = originalReadPixels.call(this, x, y, width, height, format, type, pixels);
                        // 添加轻微噪声到像素数据
                        if (pixels) {
                            for (let i = 0; i < pixels.length; i++) {
                                pixels[i] += Math.floor(Math.random() * 2);
                            }
                        }
                        return result;
                    };
                }
                
                // 模拟真实的电池状态
                Object.defineProperty(navigator, 'battery', {
                    get: () => {
                        return {
                            charging: true,
                            chargingTime: 0,
                            dischargingTime: Infinity,
                            level: 1
                        };
                    }
                });
                
                // 模拟真实的连接状态
                Object.defineProperty(navigator, 'connection', {
                    get: () => {
                        return {
                            type: 'wifi',
                            effectiveType: '4g',
                            rtt: 50,
                            downlink: 10,
                            saveData: false
                        };
                    }
                });
                
                // 模拟真实的mediaDevices - 合并了WebRTC本地IP暴露禁用功能
                Object.defineProperty(navigator, 'mediaDevices', {
                    get: () => {
                        // 创建一个基础的mediaDevices对象
                        const baseMediaDevices = {
                            enumerateDevices: async () => [
                                {deviceId: 'default', kind: 'audioinput', label: '', groupId: ''},
                                {deviceId: 'default', kind: 'videoinput', label: '', groupId: ''}
                            ],
                            getSupportedConstraints: () => ({}),
                            getUserMedia: async () => {
                                throw new Error('Permission denied');
                            },
                            getDisplayMedia: async () => {
                                throw new Error('Permission denied');
                            }
                        };
                        
                        // 添加WebRTC本地IP暴露保护
                        // 由于这是一个新创建的对象，不需要考虑原始方法的保存
                        return baseMediaDevices;
                    }
                });
                
                // 模拟真实的speechSynthesis
                Object.defineProperty(window, 'speechSynthesis', {
                    get: () => {
                        return {
                            paused: false,
                            pending: false,
                            speaking: false,
                            onvoiceschanged: null,
                            getVoices: () => [
                                {name: 'Microsoft David Desktop - English (United States)', lang: 'en-US', localService: true, default: true},
                                {name: 'Microsoft Zira Desktop - English (United States)', lang: 'en-US', localService: true, default: false}
                            ],
                            speak: () => {},
                            cancel: () => {},
                            pause: () => {},
                            resume: () => {}
                        };
                    }
                });
            """)
            
            # 添加随机延迟，模拟真实用户行为
            time.sleep(random.uniform(0.5, 1.5))
            
        elif browser_type == 'firefox':
            options = webdriver.FirefoxOptions()
            if headless:
                options.add_argument('--headless')
            
            # BotBrowser风格的浏览器指纹伪装选项
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            # 设置用户代理
            user_agent = kwargs.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0')
            options.set_preference('general.useragent.override', user_agent)
            
            # Firefox特定的指纹伪装
            options.set_preference('intl.accept_languages', kwargs.get('languages', 'en-US,en'))
            options.set_preference('media.peerconnection.enabled', False)  # 禁用WebRTC
            options.set_preference('geo.enabled', False)  # 禁用地理位置
            options.set_preference('devtools.jsonview.enabled', False)
            options.set_preference('browser.dom.window.dump.enabled', False)
            options.set_preference('extensions.webservice.discoverURL', '')
            
            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
            
            # 为Firefox添加高级反检测功能 - BotBrowser风格
            driver.execute_script("""
                // 隐藏自动化特征
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                
                // 模拟真实浏览器语言设置
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en', 'fr-FR']});
                
                // 模拟真实插件数量
                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        const plugins = [];
                        const pluginNames = ['Shockwave Flash', 'PDF Viewer', 'OpenH264 Video Codec', 'Widevine Content Decryption Module', 'NSS Internal PKCS #11 Module'];
                        const pluginDescs = ['Shockwave Flash 32.0 r0', 'Portable Document Format', 'OpenH264 Video Codec provided by Cisco Systems, Inc.', 'Widevine Content Decryption Module', 'PKCS #11 cryptographic service provider'];
                        const numPlugins = Math.floor(Math.random() * 3) + 2;
                        for (let i = 0; i < numPlugins; i++) {
                            plugins.push({
                                name: pluginNames[i % pluginNames.length],
                                description: pluginDescs[i % pluginDescs.length],
                                filename: pluginNames[i % pluginNames.length].replace(/\\s+/g, '') + '.dll'
                            });
                        }
                        return plugins;
                    }
                });
                
                // 模拟真实mime类型
                Object.defineProperty(navigator, 'mimeTypes', {
                    get: () => {
                        const mimeTypes = [];
                        const mimes = [
                            {type: 'text/html', suffixes: 'html,htm'},
                            {type: 'application/json', suffixes: 'json'},
                            {type: 'image/jpeg', suffixes: 'jpeg,jpg'},
                            {type: 'image/png', suffixes: 'png'},
                            {type: 'video/mp4', suffixes: 'mp4'},
                            {type: 'application/pdf', suffixes: 'pdf'},
                            {type: 'application/x-shockwave-flash', suffixes: 'swf'}
                        ];
                        for (const mime of mimes) {
                            mimeTypes.push(mime);
                        }
                        return mimeTypes;
                    }
                });
                
                // 修改window.navigator属性
                Object.defineProperty(navigator, 'userAgentData', {
                    get: () => {
                        return {
                            brands: [
                                {brand: 'Firefox', version: '115'},
                                {brand: 'Not A;Brand', version: '99'}
                            ],
                            platform: 'Windows',
                            platformVersion: '10.0.0',
                            architecture: 'x86',
                            model: '',
                            mobile: false
                        };
                    }
                });
                
                // 模拟真实的screen属性
                Object.defineProperty(screen, 'width', {get: () => 1920});
                Object.defineProperty(screen, 'height', {get: () => 1080});
                Object.defineProperty(screen, 'colorDepth', {get: () => 24});
                Object.defineProperty(screen, 'pixelDepth', {get: () => 24});
                Object.defineProperty(screen, 'availWidth', {get: () => 1920});
                Object.defineProperty(screen, 'availHeight', {get: () => 1040});
                Object.defineProperty(screen, 'availLeft', {get: () => 0});
                Object.defineProperty(screen, 'availTop', {get: () => 0});
                
                // 模拟真实的window属性
                Object.defineProperty(window, 'innerWidth', {get: () => 1280});
                Object.defineProperty(window, 'innerHeight', {get: () => 720});
                Object.defineProperty(window, 'outerWidth', {get: () => 1280});
                Object.defineProperty(window, 'outerHeight', {get: () => 760});
                Object.defineProperty(window, 'screenX', {get: () => 100});
                Object.defineProperty(window, 'screenY', {get: () => 50});
                Object.defineProperty(window, 'devicePixelRatio', {get: () => 1});
                
                // 模拟真实的document属性
                Object.defineProperty(document, 'visibilityState', {get: () => 'visible'});
                Object.defineProperty(document, 'hidden', {get: () => false});
                Object.defineProperty(document, 'hasFocus', {value: () => true});
                
                // 移除自动化框架痕迹
                if (window.__selenium_script__) {
                    delete window.__selenium_script__;
                }
                if (window.callPhantom) {
                    delete window.callPhantom;
                }
                if (window._phantom) {
                    delete window._phantom;
                }
                
                // 模拟真实的performance属性
                Object.defineProperty(performance, 'now', {
                    value: () => Date.now() + Math.random() * 100
                });
                
                // WebGL噪声注入 - BotBrowser风格
                if (typeof WebGLRenderingContext !== 'undefined') {
                    const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(pname) {
                        const result = originalGetParameter.call(this, pname);
                        if (pname === WebGLRenderingContext.VENDOR) {
                            return 'Google Inc.';
                        }
                        if (pname === WebGLRenderingContext.RENDERER) {
                            return 'ANGLE (NVIDIA GeForce GTX 1060 6GB Direct3D11 vs_5_0 ps_5_0)';
                        }
                        return result;
                    };
                    
                    const originalReadPixels = WebGLRenderingContext.prototype.readPixels;
                    WebGLRenderingContext.prototype.readPixels = function(x, y, width, height, format, type, pixels) {
                        const result = originalReadPixels.call(this, x, y, width, height, format, type, pixels);
                        // 添加轻微噪声到像素数据
                        if (pixels) {
                            for (let i = 0; i < pixels.length; i++) {
                                pixels[i] += Math.floor(Math.random() * 2);
                            }
                        }
                        return result;
                    };
                }
                
                // Canvas噪声注入 - BotBrowser风格（增强版）
                if (typeof HTMLCanvasElement !== 'undefined') {
                    const originalGetContext = HTMLCanvasElement.prototype.getContext;
                    HTMLCanvasElement.prototype.getContext = function(contextType, contextAttributes) {
                        const ctx = originalGetContext.call(this, contextType, contextAttributes);
                        if (ctx && contextType === '2d') {
                            // 增强的随机偏移
                            const originalFillRect = ctx.fillRect;
                            ctx.fillRect = function(x, y, width, height) {
                                const xOffset = Math.random() * 0.2 - 0.1;
                                const yOffset = Math.random() * 0.2 - 0.1;
                                originalFillRect.call(this, x + xOffset, y + yOffset, width, height);
                            };
                            
                            const originalStrokeRect = ctx.strokeRect;
                            ctx.strokeRect = function(x, y, width, height) {
                                const xOffset = Math.random() * 0.2 - 0.1;
                                const yOffset = Math.random() * 0.2 - 0.1;
                                originalStrokeRect.call(this, x + xOffset, y + yOffset, width, height);
                            };
                            
                            // 增强的像素噪声
                            const originalGetImageData = ctx.getImageData;
                            ctx.getImageData = function(sx, sy, sw, sh) {
                                const imageData = originalGetImageData.call(this, sx, sy, sw, sh);
                                
                                // 随机选择噪声模式
                                const noiseMode = Math.floor(Math.random() * 3);
                                
                                // 添加更真实的噪声到像素数据
                                for (let i = 0; i < imageData.data.length; i += 4) {
                                    let rNoise = 0, gNoise = 0, bNoise = 0;
                                    
                                    if (noiseMode === 0) {
                                        // 随机像素偏移
                                        rNoise = Math.floor(Math.random() * 3) - 1;
                                        gNoise = Math.floor(Math.random() * 3) - 1;
                                        bNoise = Math.floor(Math.random() * 3) - 1;
                                    } else if (noiseMode === 1) {
                                        // 轻微的颜色偏移
                                        const offset = Math.floor(Math.random() * 2) - 1;
                                        rNoise = offset;
                                        gNoise = offset;
                                        bNoise = offset;
                                    } else {
                                        // 区域噪声
                                        if (Math.random() < 0.1) {
                                            rNoise = Math.floor(Math.random() * 5) - 2;
                                            gNoise = Math.floor(Math.random() * 5) - 2;
                                            bNoise = Math.floor(Math.random() * 5) - 2;
                                        }
                                    }
                                    
                                    imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + rNoise));
                                    imageData.data[i + 1] = Math.max(0, Math.min(255, imageData.data[i + 1] + gNoise));
                                    imageData.data[i + 2] = Math.max(0, Math.min(255, imageData.data[i + 2] + bNoise));
                                }
                                return imageData;
                            };
                        }
                        return ctx;
                    };
                }
                
                // 模拟真实的电池状态
                Object.defineProperty(navigator, 'battery', {
                    get: () => {
                        return {
                            charging: true,
                            chargingTime: 0,
                            dischargingTime: Infinity,
                            level: 1
                        };
                    }
                });
                
                // 模拟真实的连接状态
                Object.defineProperty(navigator, 'connection', {
                    get: () => {
                        return {
                            type: 'wifi',
                            effectiveType: '4g',
                            rtt: 50,
                            downlink: 10,
                            saveData: false
                        };
                    }
                });
                
                // 模拟真实的speechSynthesis
                Object.defineProperty(window, 'speechSynthesis', {
                    get: () => {
                        return {
                            paused: false,
                            pending: false,
                            speaking: false,
                            onvoiceschanged: null,
                            getVoices: () => [
                                {name: 'Microsoft David Desktop - English (United States)', lang: 'en-US', localService: true, default: true},
                                {name: 'Microsoft Zira Desktop - English (United States)', lang: 'en-US', localService: true, default: false}
                            ],
                            speak: () => {},
                            cancel: () => {},
                            pause: () => {},
                            resume: () => {}
                        };
                    }
                });
                
                // 模拟真实的mediaDevices并禁用WebRTC本地IP暴露
                Object.defineProperty(navigator, 'mediaDevices', {
                    get: () => {
                        // 创建基础的mediaDevices对象
                        const baseMediaDevices = {
                            enumerateDevices: async () => [
                                {deviceId: 'default', kind: 'audioinput', label: '', groupId: ''},
                                {deviceId: 'default', kind: 'videoinput', label: '', groupId: ''}
                            ],
                            getSupportedConstraints: () => ({}),
                            getUserMedia: async () => {
                                throw new Error('Permission denied');
                            },
                            getDisplayMedia: async () => {
                                throw new Error('Permission denied');
                            }
                        };
                        
                        // 添加WebRTC本地IP暴露保护
                        return baseMediaDevices;
                    }
                });
            """)
            
            # 添加随机延迟，模拟真实用户行为
            time.sleep(random.uniform(0.5, 1.5))
        else:
            raise ValueError(f"不支持的浏览器类型: {browser_type}")
        
        logger.info(f"成功初始化{browser_type}浏览器")
        return driver
    
    def _click_element(self, driver, selector, wait_time=10, retries=2):
        """点击指定元素"""
        for attempt in range(retries + 1):
            try:
                element = WebDriverWait(driver, wait_time).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                
                # 模拟真实用户点击行为
                actions = ActionChains(driver)
                
                # 随机选择点击方式：直接点击或悬停后点击
                if random.random() < 0.3:
                    # 直接点击
                    x_offset = random.uniform(-5, 5)
                    y_offset = random.uniform(-5, 5)
                    actions.move_to_element_with_offset(element, x_offset, y_offset).click().perform()
                else:
                    # 悬停后点击，模拟用户思考过程
                    actions.move_to_element(element).pause(random.uniform(0.2, 0.8))
                    x_offset = random.uniform(-5, 5)
                    y_offset = random.uniform(-5, 5)
                    actions.move_by_offset(x_offset, y_offset).pause(random.uniform(0.1, 0.3)).click().perform()
                
                logger.info(f"点击了元素: {selector}")
                time.sleep(random.uniform(0.5, 1.5))  # 随机等待时间
                return
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"点击元素失败，尝试{attempt+2}/{retries+1}: {str(e)}")
                    time.sleep(random.uniform(1, 2))
                else:
                    logger.error(f"所有点击尝试都失败了: {str(e)}")
                    raise ValueError(f"点击元素失败: {str(e)}")
    
    def _fill_form(self, driver, form_data, wait_time=10):
        """填写表单"""
        for selector, value in form_data.items():
            element = WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            
            # 模拟真实用户填写表单行为
            actions = ActionChains(driver)
            
            # 随机选择是否先点击标签或其他位置，再点击输入框
            if random.random() < 0.2:
                # 先点击输入框附近位置，模拟用户定位不准确
                actions.move_to_element(element).move_by_offset(random.uniform(-20, 20), random.uniform(-20, 20))
                actions.click().pause(random.uniform(0.1, 0.3))
            
            # 点击输入框
            actions.move_to_element(element).click().perform()
            time.sleep(random.uniform(0.1, 0.5))  # 点击后等待
            
            # 随机选择是否清除内容
            if random.random() < 0.8:  # 80%的概率清除内容
                element.clear()
                time.sleep(random.uniform(0.1, 0.3))  # 清除后等待
            
            # 随机模拟输入错误并修正的情况
            if random.random() < 0.2:  # 20%的概率模拟输入错误
                # 先输入一些错误字符
                wrong_chars = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 3)))
                for char in wrong_chars:
                    element.send_keys(char)
                    time.sleep(random.uniform(0.05, 0.15))
                
                # 等待一段时间
                time.sleep(random.uniform(0.2, 0.5))
                
                # 删除错误字符
                for _ in range(len(wrong_chars)):
                    element.send_keys(Keys.BACKSPACE)
                    time.sleep(random.uniform(0.05, 0.1))
                
                # 等待一段时间
                time.sleep(random.uniform(0.1, 0.3))
            
            # 逐字符输入，模拟真实打字速度（包含随机的快速或慢速）
            for i, char in enumerate(value):
                element.send_keys(char)
                
                # 随机打字速度：偶尔有快速输入，偶尔有停顿
                if random.random() < 0.1:
                    # 10%的概率有较长停顿（模拟思考）
                    time.sleep(random.uniform(0.3, 0.8))
                elif random.random() < 0.2:
                    # 20%的概率有中等停顿
                    time.sleep(random.uniform(0.1, 0.3))
                else:
                    # 默认打字速度
                    time.sleep(random.uniform(0.03, 0.12))
                
                # 偶尔移动光标，模拟用户检查输入
                if random.random() < 0.05 and i > 0:
                    element.send_keys(Keys.LEFT)
                    time.sleep(random.uniform(0.1, 0.3))
                    element.send_keys(Keys.RIGHT)
                    time.sleep(random.uniform(0.1, 0.3))
            
            # 填写完成后，随机点击输入框外的区域
            if random.random() < 0.5:
                actions.move_to_element(element).move_by_offset(random.uniform(50, 100), random.uniform(-20, 20))
                actions.click().perform()
                time.sleep(random.uniform(0.1, 0.3))
            
            logger.info(f"在{selector}中填写了值: {value}")
            time.sleep(random.uniform(0.3, 1.2))  # 字段间随机等待
        time.sleep(random.uniform(0.5, 1.5))  # 表单填写完成后等待
    
    def _scroll_page(self, driver, scroll_count=3, max_scroll_percent=80):
        """模拟真实用户滚动页面"""
        page_height = driver.execute_script("return document.body.scrollHeight;")
        window_height = driver.execute_script("return window.innerHeight;")
        current_scroll_position = driver.execute_script("return window.pageYOffset;")
        
        for i in range(scroll_count):
            # 随机选择滚动方向：向下、向上或小幅滚动
            scroll_direction = random.choice(['down', 'up', 'small']) if current_scroll_position > window_height else 'down'
            
            if scroll_direction == 'down':
                # 向下滚动
                max_scroll_amount = (page_height - current_scroll_position - window_height) * (max_scroll_percent / 100)
                min_scroll_amount = max_scroll_amount * 0.3  # 最少滚动30%
                scroll_amount = random.uniform(min_scroll_amount, max_scroll_amount)
                new_scroll_position = current_scroll_position + scroll_amount
            elif scroll_direction == 'up':
                # 向上滚动（模拟用户回溯）
                max_scroll_amount = current_scroll_position * 0.6  # 最多向上滚动60%
                scroll_amount = random.uniform(50, max_scroll_amount)
                new_scroll_position = max(0, current_scroll_position - scroll_amount)
            else:
                # 小幅滚动（模拟用户仔细阅读）
                scroll_amount = random.uniform(-50, 100)
                new_scroll_position = max(0, min(page_height - window_height, current_scroll_position + scroll_amount))
            
            # 模拟真实滚动速度变化
            if random.random() < 0.3:
                # 缓慢滚动
                self._smooth_scroll(driver, current_scroll_position, new_scroll_position, duration=random.uniform(1, 3))
            else:
                # 正常滚动
                driver.execute_script(f"window.scrollTo(0, {new_scroll_position});")
            
            logger.info(f"页面滚动 {i+1}/{scroll_count}")
            
            # 更新当前滚动位置
            current_scroll_position = new_scroll_position
            
            # 随机等待时间，模拟用户阅读内容
            if scroll_direction == 'small':
                # 小幅滚动后等待更长时间，模拟仔细阅读
                time.sleep(random.uniform(1.5, 4))
            else:
                # 正常滚动后等待时间
                time.sleep(random.uniform(0.5, 2))
                
            # 随机模拟用户在滚动后点击页面
            if random.random() < 0.1:
                # 点击页面随机位置
                x = random.uniform(50, window_height - 50)
                y = random.uniform(50, window_height - 50)
                actions = ActionChains(driver)
                actions.move_by_offset(x, y).click().perform()
                time.sleep(random.uniform(0.3, 1))
                # 点击后返回滚动位置
                actions.move_by_offset(-x, -y).perform()
                time.sleep(random.uniform(0.1, 0.3))
                
    def _smooth_scroll(self, driver, start_pos, end_pos, duration=2):
        """平滑滚动页面"""
        start_time = driver.execute_script("return performance.now();")
        
        def ease_out_quad(t):
            return t * (2 - t)
        
        def ease_in_out_sine(t):
            return -(np.cos(np.pi * t) - 1) / 2
        
        # 随机选择缓动函数
        ease_func = random.choice([ease_out_quad, ease_in_out_sine])
        
        while True:
            current_time = driver.execute_script("return performance.now();")
            elapsed = (current_time - start_time) / 1000
            if elapsed >= duration:
                break
            
            progress = ease_func(elapsed / duration)
            new_pos = start_pos + (end_pos - start_pos) * progress
            driver.execute_script(f"window.scrollTo(0, {new_pos});")
            time.sleep(0.01)  # 避免过度占用CPU
        
        # 确保滚动到最终位置
        driver.execute_script(f"window.scrollTo(0, {end_pos});")
    
    def _extract_table_data(self, page_source, table_selector=None):
        """从页面中提取表格数据"""
        soup = BeautifulSoup(page_source, 'html.parser')
        
        if table_selector:
            table = soup.select_one(table_selector)
        else:
            table = soup.find('table')
        
        if table:
            data = pd.read_html(str(table))[0]
            return data
        return None
    
    def _extract_json_data(self, page_source):
        """从页面中提取JSON数据"""
        import re
        import json
        
        # 查找页面中的JSON数据
        json_pattern = re.compile(r'\{\s*"[a-zA-Z_]+"\s*:\s*(?:\{.*?\}|\[.*?\]|"[^"]*"|\d+|true|false|null)\s*(?:,\s*"[a-zA-Z_]+"\s*:\s*(?:\{.*?\}|\[.*?\]|"[^"]*"|\d+|true|false|null)\s*)*\}', re.DOTALL)
        json_matches = json_pattern.findall(page_source)
        
        for match in json_matches:
            try:
                json_data = json.loads(match)
                # 尝试将JSON转换为DataFrame
                if isinstance(json_data, dict):
                    # 如果是字典，尝试将其转换为DataFrame
                    for key, value in json_data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            return pd.DataFrame(value)
                    # 如果没有合适的列表，尝试直接转换字典
                    return pd.DataFrame([json_data])
                elif isinstance(json_data, list):
                    # 如果是列表，直接转换
                    return pd.DataFrame(json_data)
            except:
                continue
        return None
    
    def _extract_custom_data(self, driver, selector):
        """从自定义选择器中提取数据"""
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            # 提取数据（根据实际情况调整）
            rows = []
            for element in elements:
                text = element.text.strip()
                if text:
                    # 这里只是示例，需要根据具体网站调整数据提取逻辑
                    parts = text.split()
                    if len(parts) >= 2:
                        rows.append({'date': parts[0], 'value': float(parts[1])})
            if rows:
                return pd.DataFrame(rows)
        return None
    
    def crawl_gold9999(self, start_date=None, end_date=None, max_records=1000):
        """爬取黄金9999日线数据"""
        return self._crawl_financial_data('gold9999', start_date, end_date, max_records)
    
    def crawl_gold_london(self, start_date=None, end_date=None, max_records=1000):
        """爬取伦敦金现日线数据"""
        return self._crawl_financial_data('gold_london', start_date, end_date, max_records)
    
    def crawl_usdcny(self, start_date=None, end_date=None, max_records=1000):
        """爬取美元兑人民币汇率日线数据"""
        return self._crawl_financial_data('usdcny', start_date, end_date, max_records)
    
    def crawl_boshi_gold_c(self, start_date=None, end_date=None, max_records=1000):
        """爬取博时黄金C(002611)日线数据"""
        return self._crawl_financial_data('boshi_gold_c', start_date, end_date, max_records)
    
    def _crawl_financial_data(self, data_type, start_date=None, end_date=None, max_records=1000):
        """爬取金融数据的通用方法"""
        if data_type not in self._financial_sources:
            raise ValueError(f"不支持的金融数据源: {data_type}")
        
        source_info = self._financial_sources[data_type]
        logger.info(f"开始爬取{source_info['name']}数据")
        
        # 尝试从缓存中获取数据
        import os
        import pickle
        import hashlib
        import time
        
        # 创建缓存目录
        cache_dir = os.path.join(os.getcwd(), "data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件名
        cache_key = f"{data_type}_{max_records}"
        if start_date:
            cache_key += f"_{start_date.strftime('%Y%m%d')}"
        if end_date:
            cache_key += f"_{end_date.strftime('%Y%m%d')}"
        
        # 使用哈希值作为文件名
        cache_filename = hashlib.md5(cache_key.encode()).hexdigest() + ".pkl"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # 检查缓存是否存在且未过期（1天过期）
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            current_time = time.time()
            if current_time - cache_mtime < 86400:  # 86400秒 = 1天
                logger.info(f"从缓存中读取{source_info['name']}数据")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                logger.info(f"缓存已过期，重新爬取{source_info['name']}数据")
        
        # 尝试使用akshare作为主要数据源
        try:
            logger.info(f"尝试使用akshare获取{source_info['name']}数据")
            
            # 确保akshare库已安装
            try:
                import akshare as ak
            except ImportError:
                logger.info("正在安装akshare库...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "akshare"])
                import akshare as ak
            
            # 根据数据类型使用不同的akshare接口
            if data_type == 'gold9999':
                # 获取上海黄金交易所黄金9999数据
                # Try different akshare functions for gold data
                try:
                    # Try this function first
                    df = ak.gold_spot_price_shanghai()
                except AttributeError:
                    try:
                        # Fallback to another possible function name
                        df = ak.gold_au9999_daily()
                    except AttributeError:
                        try:
                            # Final fallback - use futures data
                            df = ak.futures_daily(symbol="AU", exchange="SHFE")
                        except AttributeError:
                            # Another fallback option
                            df = ak.stock_zh_index_daily(symbol="SHFE.AU")
                
                # 重命名列以匹配预期格式
                if '日期' in df.columns:
                    df = df.rename(columns={
                        '日期': 'date',
                        '开盘价': 'open',
                        '最高价': 'high',
                        '最低价': 'low',
                        '收盘价': 'close',
                        '成交量': 'volume',
                        '成交金额(元)': 'amount'
                    })
                elif 'date' in df.columns:
                    # 已包含正确列名，无需重命名
                    pass
                else:
                    # 处理可能的其他列名格式
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            elif data_type == 'gold_london':
                # 获取伦敦金数据 - Try different akshare functions
                try:
                    df = ak.gold_global_hist()
                except AttributeError:
                    try:
                        df = ak.gold_spot_price_kitco()
                    except AttributeError:
                        try:
                            df = ak.stock_us_daily(symbol='GC=F')
                        except AttributeError:
                            # Final fallback
                            df = ak.commodity_gold_daily()
                
                # 如果返回的是实时数据，转换为DataFrame
                if isinstance(df, dict):
                    df = pd.DataFrame([df])
                    df['date'] = pd.Timestamp.now().normalize()
                    df['open'] = df.get('price', df.get('开盘价', 0))
                    df['high'] = df.get('price', df.get('最高价', 0))
                    df['low'] = df.get('price', df.get('最低价', 0))
                    df['close'] = df.get('price', df.get('收盘价', 0))
                    df['volume'] = df.get('volume', 0)
                    df['amount'] = df.get('amount', 0)
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                else:
                    # 重命名列以匹配预期格式
                    if '日期' in df.columns:
                        df = df.rename(columns={
                            '日期': 'date',
                            '开盘价': 'open',
                            '最高价': 'high',
                            '最低价': 'low',
                            '收盘价': 'close',
                            '成交量': 'volume',
                            '成交金额': 'amount'
                        })
                    elif 'Date' in df.columns:
                        df = df.rename(columns={
                            'Date': 'date',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume',
                            'Amount': 'amount'
                        })
            elif data_type == 'usdcny':
                # 获取美元兑人民币汇率数据
                try:
                    df = ak.currency_boc_safe_exchange_rate()
                except AttributeError:
                    try:
                        df = ak.forex_usd_cny_daily()
                    except AttributeError:
                        try:
                            df = ak.stock_us_daily(symbol='USDCNY=X')
                        except AttributeError:
                            # Final fallback
                            df = ak.currency_usdcny_spot()
                
                # 处理不同数据源的格式差异
                if '货币对' in df.columns:
                    # 筛选美元兑人民币汇率
                    df = df[df['货币对'] == '美元/人民币']
                    
                    # 重命名列以匹配预期格式
                    df = df.rename(columns={
                        '日期': 'date',
                        '中间价': 'close'
                    })
                elif '日期' in df.columns:
                    df = df.rename(columns={
                        '日期': 'date',
                        '开盘价': 'open',
                        '最高价': 'high',
                        '最低价': 'low',
                        '收盘价': 'close'
                    })
                elif 'Date' in df.columns:
                    df = df.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close'
                    })
                
                # 确保所有必要列存在
                if 'open' not in df.columns:
                    df['open'] = df['close']
                if 'high' not in df.columns:
                    df['high'] = df['close']
                if 'low' not in df.columns:
                    df['low'] = df['close']
                if 'volume' not in df.columns:
                    df['volume'] = 0
                if 'amount' not in df.columns:
                    df['amount'] = 0
                
                # 选择需要的列
                df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            elif data_type == 'boshi_gold_c':
                # 获取博时黄金C(002611)基金数据 - 尝试不同的akshare函数
                # 输出akshare版本信息
                logger.info(f"akshare版本: {ak.__version__}")
                
                fund_code = "002611"
                df = None
                
                # 尝试1: 使用基金每日净值数据函数（推荐）
                try:
                    logger.info(f"尝试使用fund_open_fund_daily_em获取{source_info['name']}数据")
                    df = ak.fund_open_fund_daily_em(fund=fund_code)
                    logger.info(f"fund_open_fund_daily_em调用成功，数据形状: {df.shape}")
                    logger.info(f"数据列名: {list(df.columns)}")
                except AttributeError as e:
                    logger.warning(f"fund_open_fund_daily_em函数不存在: {str(e)}")
                except Exception as e:
                    logger.warning(f"fund_open_fund_daily_em调用失败: {str(e)}")
                
                # 尝试2: 使用另一个基金数据函数
                if df is None or df.empty:
                    try:
                        logger.info(f"尝试使用fund_em_open_fund_daily获取{source_info['name']}数据")
                        df = ak.fund_em_open_fund_daily(symbol=fund_code)
                        logger.info(f"fund_em_open_fund_daily调用成功，数据形状: {df.shape}")
                        logger.info(f"数据列名: {list(df.columns)}")
                    except AttributeError as e:
                        logger.warning(f"fund_em_open_fund_daily函数不存在: {str(e)}")
                    except Exception as e:
                        logger.warning(f"fund_em_open_fund_daily调用失败: {str(e)}")
                
                # 尝试3: 使用基金历史净值数据函数
                if df is None or df.empty:
                    try:
                        logger.info(f"尝试使用fund_net_value_history获取{source_info['name']}数据")
                        df = ak.fund_net_value_history(fund=fund_code)
                        logger.info(f"fund_net_value_history调用成功，数据形状: {df.shape}")
                        logger.info(f"数据列名: {list(df.columns)}")
                    except AttributeError as e:
                        logger.warning(f"fund_net_value_history函数不存在: {str(e)}")
                    except Exception as e:
                        logger.warning(f"fund_net_value_history调用失败: {str(e)}")
                
                # 尝试4: 使用ETF历史数据函数作为备选
                if df is None or df.empty:
                    try:
                        logger.info(f"尝试使用fund_etf_hist_em获取{source_info['name']}数据")
                        df = ak.fund_etf_hist_em(symbol=fund_code)
                        logger.info(f"fund_etf_hist_em调用成功，数据形状: {df.shape}")
                        logger.info(f"数据列名: {list(df.columns)}")
                    except AttributeError as e:
                        logger.warning(f"fund_etf_hist_em函数不存在: {str(e)}")
                    except Exception as e:
                        logger.warning(f"fund_etf_hist_em调用失败: {str(e)}")
                
                # 尝试5: 使用fund_em_open_fund_info获取基金信息
                if df is None or df.empty:
                    try:
                        logger.info(f"尝试使用fund_em_open_fund_info获取{source_info['name']}数据")
                        df = ak.fund_em_open_fund_info(symbol=fund_code)
                        logger.info(f"fund_em_open_fund_info调用成功，数据形状: {df.shape}")
                        logger.info(f"数据列名: {list(df.columns)}")
                    except AttributeError as e:
                        logger.warning(f"fund_em_open_fund_info函数不存在: {str(e)}")
                    except Exception as e:
                        logger.warning(f"fund_em_open_fund_info调用失败: {str(e)}")
                
                # 如果所有尝试都失败，抛出异常
                if df is None or df.empty:
                    logger.error(f"所有akshare函数都无法获取{source_info['name']}数据")
                    raise Exception(f"无法从akshare获取{source_info['name']}数据")
                
                # 重命名列以匹配预期格式
                logger.info(f"开始处理{source_info['name']}数据，当前列: {list(df.columns)}")
                
                # 处理不同数据源返回的列名
                if '净值日期' in df.columns:
                    if '单位净值' in df.columns:
                        df = df.rename(columns={
                            '净值日期': 'date',
                            '单位净值': 'close'
                        })
                    elif '日增长率' not in df.columns:  # 避免将增长率列作为close
                        df = df.rename(columns={
                            '净值日期': 'date',
                            df.columns[1]: 'close'  # 使用第二列作为收盘价（如果是单位净值）
                        })
                    else:
                        logger.error(f"无法识别{source_info['name']}数据的单位净值列")
                        raise Exception(f"无法识别{source_info['name']}数据的单位净值列")
                elif '日期' in df.columns:
                    if '收盘价' in df.columns:
                        df = df.rename(columns={
                            '日期': 'date',
                            '收盘价': 'close'
                        })
                    elif '单位净值' in df.columns:
                        df = df.rename(columns={
                            '日期': 'date',
                            '单位净值': 'close'
                        })
                    else:
                        df = df.rename(columns={
                            '日期': 'date',
                            df.columns[1]: 'close'  # 使用第二列作为收盘价
                        })
                elif 'Date' in df.columns:
                    df = df.rename(columns={
                        'Date': 'date',
                        'Close': 'close'
                    })
                elif '净值日期' not in df.columns and '日期' not in df.columns:
                    # 处理其他可能的日期列名
                    date_cols = [col for col in df.columns if '日期' in col or 'date' in col.lower()]
                    if date_cols:
                        df = df.rename(columns={
                            date_cols[0]: 'date',
                            df.columns[1]: 'close'  # 使用第二列作为收盘价
                        })
                    else:
                        logger.warning(f"无法识别{source_info['name']}数据的日期列，使用默认列名")
                        df.columns = ['date', 'close'] + list(df.columns[2:])
                
                # 确保所有必要列存在
                logger.info(f"处理后的数据列: {list(df.columns)}")
                
                # 复制close值到其他价格列（基金数据通常只有收盘价）
                df['open'] = df.get('open', df['close'])
                df['high'] = df.get('high', df['close'])
                df['low'] = df.get('low', df['close'])
                # 添加缺失的列
                df['volume'] = df.get('volume', 0)
                df['amount'] = df.get('amount', 0)
                
                logger.info(f"{source_info['name']}数据处理完成，数据形状: {df.shape}")
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 转换数值列
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 按日期排序
            df = df.sort_values('date')
            
            # 限制记录数量
            if max_records:
                df = df.tail(max_records)
            
            # 验证数据有效性
            if self._validate_data(df, data_type):
                logger.info(f"成功从akshare获取{source_info['name']}数据，共{len(df)}条记录")
                # 保存数据到缓存
                logger.info(f"将{source_info['name']}数据保存到缓存")
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)
                return df
            else:
                logger.error("akshare返回的数据无效")
        except Exception as e:
            logger.error(f"使用akshare获取{source_info['name']}数据失败: {str(e)}")
        
        # 尝试使用新浪财经API
        try:
            # 新浪财经的接口可能已经更新，尝试使用新的接口格式
            if data_type == 'gold9999':
                # 使用新浪财经的实时行情页面爬取
                url = "https://finance.sina.com.cn/futures/quotes/AUTD.shtml"
                
                # 添加适当的请求头和延迟
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': 'https://finance.sina.com.cn/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
                }
                
                import time
                time.sleep(1)  # 添加1秒延迟
                
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                # 新浪财经使用GBK编码，确保正确解析
                response.encoding = 'GBK'
                
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找包含历史数据的表格
                tables = soup.find_all('table')
                if tables:
                    # 尝试解析表格数据
                    for i, table in enumerate(tables):
                        try:
                            df = pd.read_html(str(table))[0]
                            logger.info(f"表格{i}形状: {df.shape}")
                            logger.info(f"表格{i}列: {list(df.columns)}")
                            logger.info(f"表格{i}前5行: {df.head()}")
                            
                            # 检查数据格式是否符合预期
                            if len(df.columns) >= 6 and df.shape[0] > 10:
                                # 假设表格有日期、开盘、最高、最低、收盘、成交量
                                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                                
                                # 转换数据类型
                                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                                df = df.dropna(subset=['date'])
                                
                                for col in ['open', 'high', 'low', 'close', 'volume']:
                                    if col in df.columns:
                                        df[col] = pd.to_numeric(df[col], errors='coerce')
                                
                                # 按日期排序
                                df = df.sort_values('date')
                                
                                # 限制记录数量
                                if max_records:
                                    df = df.tail(max_records)
                                
                                # 验证数据有效性
                                if self._validate_data(df, data_type):
                                    logger.info(f"成功从网页表格爬取{source_info['name']}数据，共{len(df)}条记录")
                                    # 保存数据到缓存
                                    logger.info(f"将{source_info['name']}数据保存到缓存")
                                    with open(cache_path, 'wb') as f:
                                        pickle.dump(df, f)
                                    return df
                                else:
                                    logger.error("网页表格爬取的数据无效")
                        except Exception as e:
                            logger.warning(f"解析表格{i}失败: {str(e)}")
                            continue
            
            # 如果表格解析失败，尝试使用东方财富网的API（备选数据源）
            logger.info(f"尝试使用东方财富网API获取{source_info['name']}数据")
            # 更新东方财富网API端点和参数，使用更稳定的接口
            base_url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
            
            # 根据数据类型设置正确的secid和其他参数
            if data_type == 'gold9999':
                # 上海黄金交易所黄金9999 (AUTD) - 使用正确的secid
                secid = "8.000060"
            elif data_type == 'gold_london':
                # 伦敦金现 (XAUUSD)
                secid = "100.XAUUSD"
            elif data_type == 'usdcny':
                # 美元兑人民币汇率 (USDCNY)
                secid = "100.USDCNY"
            elif data_type == 'boshi_gold_c':
                # 博时黄金C (002611)
                secid = "0.002611"
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")
            
            # 构建完整的API请求URL，添加更多参数以确保获取数据
            params = {
                'secid': secid,
                'fields1': 'f1,f2,f3,f4,f5',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
                'klt': '101',  # 日线数据
                'fqt': '1',    # 复权类型
                'beg': '0',    # 开始时间，0表示从最开始
                'end': '20500101'  # 结束时间，设置为未来日期以获取最新数据
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'http://quote.eastmoney.com/',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'X-Requested-With': 'XMLHttpRequest',
                'Origin': 'http://quote.eastmoney.com',
                'Connection': 'keep-alive'
            }
            
            import time
            time.sleep(1)  # 添加1秒延迟
            
            # 使用params参数构建URL
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            
            import json
            data = response.json()
            
            if data['data'] and data['data']['klines']:
                klines = data['data']['klines']
                data_list = [line.split(',') for line in klines]
                
                df = pd.DataFrame(data_list)
                df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'change_rate']
                
                # 转换数据类型
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 按日期排序
                df = df.sort_values('date')
                
                # 限制记录数量
                if max_records:
                    df = df.tail(max_records)
                
                # 验证数据有效性
                if self._validate_data(df, data_type):
                    logger.info(f"成功从东方财富网API爬取{source_info['name']}数据，共{len(df)}条记录")
                    # 保存数据到缓存
                    logger.info(f"将{source_info['name']}数据保存到缓存")
                    with open(cache_path, 'wb') as f:
                        pickle.dump(df, f)
                    return df
                else:
                    logger.error("东方财富网API返回的数据无效")
        except Exception as e:
            logger.error(f"使用新浪财经和东方财富网API获取{source_info['name']}数据失败: {str(e)}")
        
        # 尝试使用雅虎财经API（添加了请求头和延迟）
        try:
            logger.info(f"尝试使用雅虎财经API获取{source_info['name']}数据")
            
            # 检查yfinance库是否已安装
            try:
                import yfinance as yf
            except ImportError:
                logger.info("正在安装yfinance库...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
                import yfinance as yf
            
            # 雅虎财经的代码映射
            ticker_map = {
                'gold9999': 'GC=F',  # COMEX黄金期货
                'gold_london': 'GC=F',  # 使用COMEX黄金期货作为伦敦金的替代
                'usdcny': 'CNY=X'  # 美元兑人民币汇率
            }
            
            if data_type in ticker_map:
                ticker = ticker_map[data_type]
                
                # 获取历史数据
                start_date = pd.Timestamp.now() - pd.Timedelta(days=max_records)
                end_date = pd.Timestamp.now()
                
                # 添加请求头和延迟
                import time
                time.sleep(1.5)  # 添加更长的延迟避免被限速
                
                # 设置yfinance的请求头
                yf.pdr_override()
                
                # 尝试使用yfinance获取数据，添加更多参数
                df = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    threads=False,
                    group_by='ticker',
                    progress=False,
                    auto_adjust=False,
                    actions=False
                )
                
                if not df.empty:
                    # 转换为与其他数据源一致的格式
                    df = df.reset_index()
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                    
                    # 只保留需要的列
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    
                    # 添加amount列（雅虎财经没有直接提供）
                    df['amount'] = df['close'] * df['volume']
                    
                    # 限制记录数量
                    if max_records:
                        df = df.tail(max_records)
                    
                    # 验证数据有效性
                    if self._validate_data(df, data_type):
                        logger.info(f"成功从雅虎财经API爬取{source_info['name']}数据，共{len(df)}条记录")
                        # 保存数据到缓存
                        logger.info(f"将{source_info['name']}数据保存到缓存")
                        with open(cache_path, 'wb') as f:
                            pickle.dump(df, f)
                        return df
                    else:
                        logger.error("雅虎财经API返回的数据无效")
                else:
                    logger.error("雅虎财经API返回数据为空")
        except Exception as e:
            logger.error(f"使用雅虎财经API获取{source_info['name']}数据失败: {str(e)}")
        
        # 尝试使用英为财情API
        try:
            logger.info(f"尝试使用英为财情API获取{source_info['name']}数据")
            
            # 英为财情的API接口
            if data_type == 'gold9999':
                url = "https://www.investing.com/instruments/HistoricalDataAjax"
                params = {
                    'curr_id': '8830',  # 黄金现货
                    'smlID': '1159963',
                    'header': 'Gold%20Spot',
                    'st_date': (pd.Timestamp.now() - pd.Timedelta(days=max_records)).strftime('%m/%d/%Y'),
                    'end_date': pd.Timestamp.now().strftime('%m/%d/%Y'),
                    'interval_sec': 'Daily',
                    'sort_col': 'date',
                    'sort_ord': 'DESC',
                    'action': 'historical_data'
                }
            elif data_type == 'gold_london':
                url = "https://www.investing.com/instruments/HistoricalDataAjax"
                params = {
                    'curr_id': '8830',  # 黄金现货
                    'smlID': '1159963',
                    'header': 'Gold%20Spot',
                    'st_date': (pd.Timestamp.now() - pd.Timedelta(days=max_records)).strftime('%m/%d/%Y'),
                    'end_date': pd.Timestamp.now().strftime('%m/%d/%Y'),
                    'interval_sec': 'Daily',
                    'sort_col': 'date',
                    'sort_ord': 'DESC',
                    'action': 'historical_data'
                }
            elif data_type == 'usdcny':
                url = "https://www.investing.com/instruments/HistoricalDataAjax"
                params = {
                    'curr_id': '151',  # 美元兑人民币汇率
                    'smlID': '1159963',
                    'header': 'USD/CNY',
                    'st_date': (pd.Timestamp.now() - pd.Timedelta(days=max_records)).strftime('%m/%d/%Y'),
                    'end_date': pd.Timestamp.now().strftime('%m/%d/%Y'),
                    'interval_sec': 'Daily',
                    'sort_col': 'date',
                    'sort_ord': 'DESC',
                    'action': 'historical_data'
                }
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://www.investing.com/',
                'X-Requested-With': 'XMLHttpRequest'
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # 解析HTML响应
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='historicalTbl')
            
            if table:
                    df = pd.read_html(str(table))[0]
                    
                    # 重命名列
                    df.columns = ['date', 'close', 'open', 'high', 'low', 'change_percent']
                    
                    # 转换日期格式
                    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
                    
                    # 转换数值列
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
                    
                    # 排序并限制记录数量
                    df = df.sort_values('date')
                    if max_records:
                        df = df.tail(max_records)
                    
                    # 添加volume和amount列（英为财情没有直接提供）
                    df['volume'] = 0
                    df['amount'] = df['close'] * df['volume']
                    
                    # 验证数据有效性
                    if self._validate_data(df, data_type):
                        logger.info(f"成功从英为财情API爬取{source_info['name']}数据，共{len(df)}条记录")
                        # 保存数据到缓存
                        logger.info(f"将{source_info['name']}数据保存到缓存")
                        with open(cache_path, 'wb') as f:
                            pickle.dump(df, f)
                        return df
                    else:
                        logger.error("英为财情API返回的数据无效")
            else:
                logger.error("英为财情API没有找到数据表格")
        except Exception as e:
            logger.error(f"使用英为财情API获取{source_info['name']}数据失败: {str(e)}")
        
        # 如果所有API都失败，抛出异常而不是生成模拟数据
        raise Exception(f"所有数据源（akshare、新浪财经、东方财富网、雅虎财经、英为财情）均无法获取{source_info['name']}的真实数据，请检查网络连接和API可用性")
    
    def clean_data(self, drop_na=True, method='ffill', threshold=3):
        """数据清洗：处理缺失值和异常值"""
        if drop_na:
            # 删除缺失值
            self.data = self.data.dropna(subset=[self.target_column])
        else:
            # 填充缺失值
            if method == 'ffill':
                self.data[self.target_column] = self.data[self.target_column].ffill()
            elif method == 'bfill':
                self.data[self.target_column] = self.data[self.target_column].bfill()
            elif method == 'mean':
                self.data[self.target_column] = self.data[self.target_column].fillna(
                    self.data[self.target_column].mean())
            elif method == 'median':
                self.data[self.target_column] = self.data[self.target_column].fillna(
                    self.data[self.target_column].median())
        
        # 处理异常值（使用Z-score方法）
        z_scores = np.abs((self.data[self.target_column] - self.data[self.target_column].mean()) / 
                         self.data[self.target_column].std())
        self.data = self.data[z_scores < threshold]
        
        return self.data
    
    def _validate_data(self, df, data_type):
        """验证爬取的数据是否有效"""
        if df is None or df.empty:
            logger.error(f"{data_type}数据为空")
            return False
        
        # 检查必要的列是否存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"{data_type}数据缺少必要的列: {col}")
                return False
        
        # 检查日期范围
        valid_dates = df['date'].dropna()
        if len(valid_dates) < 10:
            logger.error(f"{data_type}数据日期数量不足，仅{len(valid_dates)}条有效日期")
            return False
        
        # 检查日期顺序
        if not df['date'].is_monotonic_increasing:
            logger.warning(f"{data_type}数据日期顺序不正确，正在重新排序")
            df = df.sort_values('date')
        
        # 检查价格是否在合理范围内
        if data_type == 'gold9999':
            # 黄金9999价格范围（元/克）
            min_price, max_price = 300, 600
        elif data_type == 'gold_london':
            # 伦敦金价格范围（美元/盎司）
            min_price, max_price = 1000, 3000
        elif data_type == 'usdcny':
            # 美元兑人民币汇率范围
            min_price, max_price = 6.0, 8.0
        elif data_type == 'boshi_gold_c':
            # 博时黄金C基金净值范围（根据历史数据）
            min_price, max_price = 1.0, 5.0
        else:
            min_price, max_price = 0, float('inf')
        
        # 检查收盘价是否在合理范围内
        if data_type in ['gold9999', 'gold_london', 'usdcny', 'boshi_gold_c']:
            if df['close'].min() < min_price or df['close'].max() > max_price:
                logger.warning(f"{data_type}数据价格超出合理范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
                # 移除价格超出范围的异常值
                df = df[(df['close'] >= min_price) & (df['close'] <= max_price)]
                if df.empty:
                    logger.error(f"移除异常值后{data_type}数据为空")
                    return False
        
        # 检查价格数据是否为数字
        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"{data_type}数据中{col}列不是数字类型")
                return False
        
        # 检查价格数据是否有缺失值
        for col in ['open', 'high', 'low', 'close']:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logger.warning(f"{data_type}数据中{col}列有{missing_count}个缺失值，正在填充")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # 检查成交量和成交额是否为非负
        for col in ['volume', 'amount']:
            if df[col].min() < 0:
                logger.warning(f"{data_type}数据中{col}列包含负值，正在修正")
                df[col] = df[col].abs()
        
        # 检查数据是否有重复的日期
        if df['date'].duplicated().any():
            logger.warning(f"{data_type}数据包含重复日期，正在去重")
            df = df.drop_duplicates(subset=['date'], keep='last')
        
        # 检查数据的完整性（连续日期）
        if data_type == 'boshi_gold_c':
            # 对于基金数据，检查日期连续性
            df = df.set_index('date')
            expected_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            missing_dates = expected_dates.difference(df.index)
            if len(missing_dates) > 0:
                logger.warning(f"{data_type}数据缺少{len(missing_dates)}个交易日")
            df = df.reset_index()
        
        # 检查价格波动是否合理
        if len(df) > 1:
            df['price_change_pct'] = df['close'].pct_change() * 100
            max_daily_change = df['price_change_pct'].abs().max()
            if max_daily_change > 10:  # 单日波动超过10%视为异常
                logger.warning(f"{data_type}数据单日最大波动{max_daily_change:.2f}%，可能存在异常值")
        
        # 检查数据量是否足够
        if len(df) < 30:
            logger.warning(f"{data_type}数据量较少，仅{len(df)}条记录")
        
        logger.info(f"{data_type}数据验证通过，最终数据量: {len(df)}条")
        return True
    
    def check_stationarity(self):
        """检验时间序列的平稳性"""
        result = adfuller(self.data[self.target_column])
        print('ADF统计量:', result[0])
        print('p-value:', result[1])
        print('临界值:', result[4])
        
        if result[1] <= 0.05:
            print('结论：序列是平稳的（拒绝原假设）')
            return True
        else:
            print('结论：序列是非平稳的（无法拒绝原假设）')
            return False
    
    def make_stationary(self, method='diff', order=1):
        """使时间序列平稳化"""
        if method == 'diff':
            # 差分
            self.data[f'{self.target_column}_diff'] = self.data[self.target_column].diff(periods=order)
            self.data.dropna(inplace=True)
            self.target_column = f'{self.target_column}_diff'
        elif method == 'log':
            # 对数变换
            self.data[f'{self.target_column}_log'] = np.log(self.data[self.target_column])
            self.target_column = f'{self.target_column}_log'
        elif method == 'sqrt':
            # 平方根变换
            self.data[f'{self.target_column}_sqrt'] = np.sqrt(self.data[self.target_column])
            self.target_column = f'{self.target_column}_sqrt'
        
        return self.data
    
    def normalize(self, method='min-max'):
        """数据归一化"""
        if method == 'min-max':
            min_val = self.data[self.target_column].min()
            max_val = self.data[self.target_column].max()
            self.data[f'{self.target_column}_norm'] = (self.data[self.target_column] - min_val) / (max_val - min_val)
        elif method == 'z-score':
            mean_val = self.data[self.target_column].mean()
            std_val = self.data[self.target_column].std()
            self.data[f'{self.target_column}_norm'] = (self.data[self.target_column] - mean_val) / std_val
        
        return self.data
    
    def visualize_time_series(self, title='时间序列图', figsize=(12, 6)):
        """绘制时间序列图"""
        plt.figure(figsize=figsize)
        plt.plot(self.data.index, self.data[self.target_column], label=self.target_column)
        plt.title(title)
        plt.xlabel('时间')
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def visualize_trend(self, window=30, title='趋势图', figsize=(12, 6)):
        """绘制趋势图"""
        self.data['rolling_mean'] = self.data[self.target_column].rolling(window=window).mean()
        self.data['rolling_std'] = self.data[self.target_column].rolling(window=window).std()
        
        plt.figure(figsize=figsize)
        plt.plot(self.data.index, self.data[self.target_column], label='原始数据')
        plt.plot(self.data.index, self.data['rolling_mean'], label=f'{window}天移动平均', color='red')
        plt.plot(self.data.index, self.data['rolling_std'], label=f'{window}天标准差', color='green')
        plt.title(title)
        plt.xlabel('时间')
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def visualize_seasonality(self, freq='M', title='季节性图', figsize=(12, 6)):
        """绘制季节性图"""
        # 按指定频率重采样并计算平均值
        seasonal_data = self.data[self.target_column].resample(freq).mean()
        
        plt.figure(figsize=figsize)
        plt.plot(seasonal_data.index, seasonal_data, marker='o')
        plt.title(title)
        plt.xlabel('时间')
        plt.ylabel(self.target_column)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()
    
    def visualize_histogram(self, bins=20, title='数据分布直方图', figsize=(10, 6)):
        """绘制数据分布直方图"""
        plt.figure(figsize=figsize)
        sns.histplot(self.data[self.target_column], bins=bins, kde=True)
        plt.title(title)
        plt.xlabel(self.target_column)
        plt.ylabel('频率')
        plt.grid(True)
        plt.show()
    
    def split_data(self, train_size=0.8):
        """划分训练集和测试集"""
        split_point = int(len(self.data) * train_size)
        train_data = self.data[:split_point]
        test_data = self.data[split_point:]
        
        return train_data, test_data
    
    def get_data(self):
        """获取当前数据"""
        return self.data
    
    def get_info(self):
        """获取数据信息"""
        print("数据基本信息:")
        print(self.data.info())
        print("\n数据统计描述:")
        print(self.data.describe())
        print(f"\n时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
        print(f"数据点数量: {len(self.data)}")
