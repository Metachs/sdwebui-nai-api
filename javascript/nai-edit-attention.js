function keyupEditAttentionNAI(event) {
  if (!opts.nai_keyedit_attention) return;
  let target = event.originalTarget || event.composedPath()[0];
  if (!target.matches("*:is([id*='_toprow'] [id*='_prompt'], .prompt) textarea")) return;
  if (!(event.altKey)) return;

  let isPlus = event.key == "ArrowUp";
  let isMinus = event.key == "ArrowDown";
  if (!isPlus && !isMinus) return;

  let selectionStart = target.selectionStart;
  let selectionEnd = target.selectionEnd;
  let text = target.value;
  if (!(text.length > 0)) return;

  function selectCurrentWord() {
    if (selectionStart !== selectionEnd) return false;
    const whitespace_delimiters = {
      "Tab": "\t",
      "Carriage Return": "\r",
      "Line Feed": "\n",
    };
    let delimiters = opts.keyedit_delimiters;

    for (let i of opts.keyedit_delimiters_whitespace) {
      delimiters += whitespace_delimiters[i];
    }

    // seek backward to find beginning
    while (
      !delimiters.includes(text[selectionStart - 1]) &&
      selectionStart > 0
    ) {
      selectionStart--;
    }

    // seek forward to find end
    while (
      !delimiters.includes(text[selectionEnd]) &&
      selectionEnd < text.length
    ) {
      selectionEnd++;
    }

    // deselect surrounding whitespace
    while (
      target.textContent.slice(selectionStart, selectionStart + 1) == " " &&
      selectionStart < selectionEnd
    ) {
      selectionStart++;
    }
    while (
      target.textContent.slice(selectionEnd - 1, selectionEnd) == " " &&
      selectionEnd > selectionStart
    ) {
      selectionEnd--;
    }

    target.setSelectionRange(selectionStart, selectionEnd);
    return true;
  }

  selectCurrentWord();

  event.preventDefault();

  const start = selectionStart > 0 ? text[selectionStart - 1] : "";
  const end = text[selectionEnd];
  const deltaCurrent = !["{", "["].includes(start) ? 0 : start == "{" ? 1 : -1;
  const deltaUser = isPlus ? 1 : -1;
  let selectionStartDelta = 0;
  let selectionEndDelta = 0;

  function addBrackets(str, isPlus) {
    if (isPlus) {
      str = `{${str}}`;
    } else {
      str = `[${str}]`;
    }
    return str;
  }

  /* modify text */
  let modifiedText = text.slice(selectionStart, selectionEnd);
  if (deltaCurrent == 0 || deltaCurrent == deltaUser) {
    modifiedText = addBrackets(modifiedText, isPlus);
    selectionStartDelta += 1;
    selectionEndDelta += 1;
  } else {
    selectionStart--;
    selectionEnd++;
    selectionEndDelta -= 2;
  }

  text = text.slice(0, selectionStart) + modifiedText + text.slice(selectionEnd);

  target.focus();
  target.value = text;
  target.selectionStart = selectionStart + selectionStartDelta;
  target.selectionEnd = selectionEnd + selectionEndDelta;

  updateInput(target);
}

addEventListener("keydown", (event) => {
  keyupEditAttentionNAI(event);
});
