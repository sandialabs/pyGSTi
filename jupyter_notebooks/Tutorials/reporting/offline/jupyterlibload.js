//loads libraries from offline resources the jupyter notebook HTML 
// files just load from CDN (assuming an internet connection).
// Note: could load a local MathJax here also, but don't bother
//  with this for now (MathJax is big, and we don't really need it).
if(typeof requirejs === "undefined") {
    document.write("<script src='offline/require.min.js'><\/script>");
    console.log("Loading offline requireJS");
}
if(typeof window.jQuery === "undefined") {
    document.write("<script src='offline/jquery-3.6.4.min.js'><\/script>");
    console.log("Loading offline jQuery");
}
