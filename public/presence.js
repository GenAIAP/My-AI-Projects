(() => {
    const identityKey = 'site-presence-client-id';
    let clientId = localStorage.getItem(identityKey);
    if (!clientId) {
        clientId = typeof crypto.randomUUID === 'function'
            ? crypto.randomUUID()
            : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        localStorage.setItem(identityKey, clientId);
    }

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
