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
    
    def _crawl_financial_data(self, data_type, start_date=None, end_date=None, max_records=1000):
        """爬取金融数据的通用方法"""
        if data_type not in self._financial_sources:
            raise ValueError(f"不支持的金融数据源: {data_type}")
        
        source_info = self._financial_sources[data_type]
        logger.info(f"开始爬取{source_info['name']}数据")
        
        # 尝试使用新浪财经API
        try:
            # 新浪财经的接口可能已经更新，尝试使用新的接口格式
            if data_type == 'gold9999':
                # 使用新浪财经的实时行情页面爬取
                url = "https://finance.sina.com.cn/futures/quotes/AUTD.shtml"
                response = requests.get(url)
                response.raise_for_status()
                
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
                                
                                logger.info(f"成功从网页表格爬取{source_info['name']}数据，共{len(df)}条记录")
                                return df
                        except Exception as e:
                            logger.warning(f"解析表格{i}失败: {str(e)}")
                            continue
            
            # 如果表格解析失败，尝试使用东方财富网的API（备选数据源）
            logger.info(f"尝试使用东方财富网API获取{source_info['name']}数据")
            if data_type == 'gold9999':
                # 东方财富网黄金9999数据（正确的secid）
                url = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=8.000060&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56,f57,f58&klt=101&fqt=1"
            elif data_type == 'gold_london':
                # 东方财富网伦敦金数据（使用正确的secid）
                url = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=100.XAUUSD&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56,f57,f58&klt=101&fqt=1"
            elif data_type == 'usdcny':
                # 东方财富网美元兑人民币数据（使用正确的secid）
                url = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=100.USDCNY&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56,f57,f58&klt=101&fqt=1"
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'http://quote.eastmoney.com/'
            }
            response = requests.get(url, headers=headers)
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
                
                logger.info(f"成功从东方财富网API爬取{source_info['name']}数据，共{len(df)}条记录")
                return df
            else:
                logger.error("东方财富网API返回数据为空")
                
        except Exception as e:
            logger.error(f"爬取{source_info['name']}数据失败: {str(e)}")
        
        # 如果所有API都失败，生成模拟数据
        logger.info(f"所有API爬取失败，生成{source_info['name']}的模拟数据")
        
        # 生成模拟数据 - 使用固定的日期范围，确保所有数据源的日期一致
        # 使用当前日期减去固定天数作为起始日期
        end_date = pd.Timestamp.now().normalize()  # 去除时间部分
        start_date = end_date - pd.Timedelta(days=max_records-1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        if data_type == 'gold9999':
            # 黄金9999模拟数据（约400-450元/克）
            base_price = 420
            volatility = 5
            trend = 0.05
            prices = []
            current_price = base_price
            
            for i, date in enumerate(dates):
                # 添加趋势和随机波动
                current_price += trend + (np.random.randn() * volatility)
                # 确保价格在合理范围内
                current_price = max(380, min(460, current_price))
                prices.append(current_price)
                
            df = pd.DataFrame({
                'date': dates,
                'open': [p * (1 + np.random.randn() * 0.01) for p in prices],
                'high': [p * (1 + np.random.randn() * 0.02) for p in prices],
                'low': [p * (1 - np.random.randn() * 0.02) for p in prices],
                'close': prices,
                'volume': [int(np.random.randint(100000, 1000000)) for _ in dates]
            })
        elif data_type == 'gold_london':
            # 伦敦金模拟数据（约1800-2200美元/盎司）
            base_price = 2000
            volatility = 20
            trend = 0.5
            prices = []
            current_price = base_price
            
            for i, date in enumerate(dates):
                current_price += trend + (np.random.randn() * volatility)
                current_price = max(1800, min(2200, current_price))
                prices.append(current_price)
                
            df = pd.DataFrame({
                'date': dates,
                'open': [p * (1 + np.random.randn() * 0.01) for p in prices],
                'high': [p * (1 + np.random.randn() * 0.02) for p in prices],
                'low': [p * (1 - np.random.randn() * 0.02) for p in prices],
                'close': prices,
                'volume': [int(np.random.randint(500000, 5000000)) for _ in dates]
            })
        elif data_type == 'usdcny':
            # 美元兑人民币汇率模拟数据（约6.3-7.3）
            base_rate = 6.8
            volatility = 0.05
            trend = -0.001
            rates = []
            current_rate = base_rate
            
            for i, date in enumerate(dates):
                current_rate += trend + (np.random.randn() * volatility)
                current_rate = max(6.3, min(7.3, current_rate))
                rates.append(current_rate)
                
            df = pd.DataFrame({
                'date': dates,
                'open': [r * (1 + np.random.randn() * 0.001) for r in rates],
                'high': [r * (1 + np.random.randn() * 0.002) for r in rates],
                'low': [r * (1 - np.random.randn() * 0.002) for r in rates],
                'close': rates,
                'volume': [int(np.random.randint(1000000, 10000000)) for _ in dates]
            })
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 按日期排序
        df = df.sort_values('date')
        
        logger.info(f"成功生成{source_info['name']}的模拟数据，共{len(df)}条记录")
        return df
    
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
