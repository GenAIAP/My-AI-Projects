(() => {
    const identityKey = 'site-presence-client-id';
    const cookieKey = 'site_presence_client_id';
    const cookie = document.cookie.split(';').map(value => value.trim())
        .find(value => value.startsWith(`${cookieKey}=`));
    let clientId = localStorage.getItem(identityKey);
    if (!clientId && cookie) {
        try {
            clientId = decodeURIComponent(cookie.slice(cookieKey.length + 1));
        } catch {
            clientId = null;
        }
    }
    if (!clientId) {
        clientId = typeof crypto.randomUUID === 'function'
            ? crypto.randomUUID()
            : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    }
    localStorage.setItem(identityKey, clientId);
    const secureCookie = window.location.protocol === 'https:' ? '; Secure' : '';
    document.cookie = `${cookieKey}=${encodeURIComponent(clientId)}; Path=/; Max-Age=31536000; SameSite=Lax${secureCookie}`;

    window.presenceSocketOptions = () => ({
        auth: {
            clientId,
            name: localStorage.getItem('chat-user-name') || ''
        }
    });

    window.watchForPresenceBans = (socket) => {
        let refreshPending = false;
        const refreshBannedPage = () => {
            if (refreshPending) return;
            refreshPending = true;
            window.location.reload();
        };

        socket.on('ban', refreshBannedPage);
        socket.on('connect_error', (error) => {
            if (error.data?.code === 'USER_BANNED') refreshBannedPage();
        });
    };
})();
