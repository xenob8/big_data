function FindProxyForURL(url, host) {

    if (shExpMatch(host, "2ip.ru")) {
        return "HTTPS 77.221.147.153:55555";
    }

    if (shExpMatch(host, "discordapp.com")) {
        return "HTTPS 77.221.147.153:55555";
    }

    return "DIRECT";
}
