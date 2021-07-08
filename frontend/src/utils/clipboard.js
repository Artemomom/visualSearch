export function retrieveImageFromClipboardAsBlob(pasteEvent, callback) {
  if (pasteEvent.clipboardData === false) {
    if (typeof callback === 'function') {
      callback(undefined);
    }
  }

  const items = pasteEvent.clipboardData.items;

  if (items === undefined) {
    if (typeof callback === 'function') {
      callback(undefined);
    }
  }

  for (let i = 0; i < items.length; i++) {
    // Skip content if not image
    // eslint-disable-next-line no-continue
    if (items[i].type.indexOf('image') === -1) continue;
    // Retrieve image on clipboard as blob
    const blob = items[i].getAsFile();

    if (typeof callback === 'function') {
      callback(blob);
    }
  }
}
