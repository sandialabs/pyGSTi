function show(shown, hidden) {
  document.getElementById(shown).style.display='block';
  document.getElementById(hidden).style.display='none';
  return false;
}

function toggle(shown, hidden) {
    var x = document.getElementsByClassName(shown);
    var i;
    for (i = 0; i < x.length; i++) {
            x[i].style.display='block';
    } 
    x = document.getElementsByClassName(hidden);
    for (i = 0; i < x.length; i++) {
            x[i].style.display='none';
    } 
}
