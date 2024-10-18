function FindProxyForURL(url, host) {

    if (shExpMatch(host, "*.2ip.ru")) {
        return "PROXY 77.221.147.153:55555";
    }

    return "DIRECT";
}
