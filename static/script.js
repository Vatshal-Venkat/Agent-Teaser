function scrollChatToBottom(){
    try {
        const containers = document.getElementsByClassName('chat-container');
        if(containers && containers.length>0){
            const container = containers[0];
            container.scrollTop = container.scrollHeight;
        }
    } catch(e) { console.error(e); }
}

function copyToClipboard(elementId) {
    try {
        const element = document.getElementById(elementId);
        if (!element) {
            alert("Error: Could not find text to copy");
            return;
        }
        const text = element.innerText;
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(
                () => { alert("Copied to clipboard!"); },
                (err) => { alert("Copy failed: " + err); }
            );
        } else {
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            alert('Copied (fallback)');
        }
    } catch(err) {
        alert("Error copying text: " + err);
    }
}