<!DOCTYPE html><html lang=""><head><meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width,maximum-scale=1,user-scalable=0,initial-scale=1"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="Cache-Control" content="no-cache, must-revalidate"><meta http-equiv="expires" content="0"><link rel="icon" href="/uuapstatic/img/ph-logo.5e92fca2.svg"><title>统一认证中心</title><script>function ieBrowser() {
            var userAgent = navigator.userAgent; //取得浏览器的userAgent字符串
            var isIE = userAgent.indexOf("compatible") > -1 && userAgent.indexOf("MSIE") > -1; //判断是否IE<11浏览器
            var isIE11 = userAgent.indexOf('Trident') > -1 && userAgent.indexOf("rv:11.0") > -1;
            return (isIE || isIE11) ? true : false;
        }
        window.onload = function() {
            var isIE = ieBrowser();
            if (isIE) {
                var nav = window.navigator;
                var isZh = nav.userLanguage.indexOf('zh') !== -1 && nav.systemLanguage.indexOf('zh') !== -1 && nav.browserLanguage.indexOf('zh') !== -1;
                document.getElementById(isZh ? 'enLang' : 'chLang').style.display = 'none';
                document.getElementById('too-low').style.display = 'block';
            }
        };</script><link href="/uuapstatic/css/app.890c08bc.css" rel="preload" as="style"><link href="/uuapstatic/css/chunk-vendors.ec2db238.css" rel="preload" as="style"><link href="/uuapstatic/js/app.2b07e65a.js" rel="preload" as="script"><link href="/uuapstatic/js/chunk-vendors.99baec38.js" rel="preload" as="script"><link href="/uuapstatic/css/chunk-vendors.ec2db238.css" rel="stylesheet"><link href="/uuapstatic/css/app.890c08bc.css" rel="stylesheet"></head><body><noscript>您的浏览器不支持或禁止了网页脚本(JavaScript)，请在浏览器设置中允许运行JavaScript，如有问题请联系如流群：1667521</noscript><div id="too-low" class="too-low" style="display: none;"><div class="header"><img class="cursor-pointer" src="/uuapstatic/logo.svg" alt="logo" height="38"></div><div class="too-low-con"><div><img src="/uuapstatic/error-auth.png" alt="error-auth.png"><br><br><div id="enLang">Your browser version is too old. In order to have a better access experience, please change your browser to firebox,<br>Safari, chrome (including chrome kernel browser) and edge. Chrome is recommended.<br><br>Browser compatible version:</div><div id="chLang">您的浏览器版本过旧，为了更好的访问体验，请更换浏览器为firefox，Safari，<br>Chrome（包含Chromium内核浏览器）及edge，推荐使用Chrome。<br><br>浏览器兼容版本：</div>firefox(58+), safari(10+), chrome(76+), edge(16+)</div></div></div><div id="uuap"></div><script src="/uuapstatic/js/chunk-vendors.99baec38.js"></script><script src="/uuapstatic/js/app.2b07e65a.js"></script></body><script src="https://wappass.baidu.com/static/machine/js/api/mkd.js"></script><script>function initQRCode({appToken, language}) {
        return new Promise((resolve, reject) => {
            new Beep_QRCode({
                el: 'qrCode',
                // appToken 请选择颁发的线上appToken
                appToken,
                language,
                success: function (res) {
                    resolve(res);
                },
                error: function (res) {
                    reject(res);
                }
            });
        });
    }</script></html>