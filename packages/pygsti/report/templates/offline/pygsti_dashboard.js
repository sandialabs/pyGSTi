
sidenav_closed_width = 10;
sidenav_mousetrigger_width = 50;
sidenav_width = 200;

//Maybe need this instead (for Katex?) document.addEventListener("DOMContentLoaded", function() {
$(document).ready(function() {

    window.plotman = new PlotManager();
    window.navtacked = true;
    openNav();

    document.getElementById("defaultOpen").click();
    
    $("body").on("mousemove",function(event) {
	if(!window.navtacked) {
	    if (event.pageX < sidenav_mousetrigger_width) {
		openNav();
	    }
	    else if (event.pageX > sidenav_width) {
		closeNav();
	    }
	}
    });
    
    //Top banner/header JS below here
    var elSelector		= '.header',
	elClassHidden	= 'header--hidden',
	throttleTimeout	= 500,
	$element		= $( elSelector );
    
    if( !$element.length ) return true;
    
    var $window			= $( window ),
	wHeight			= 0,
	wScrollCurrent	= 0,
	wScrollBefore	= 0,
	wScrollDiff		= 0,
	$document		= $( document ),
	dHeight			= 0,
	
	throttle = function( delay, fn ){
	    var last, deferTimer;
	    return function() {
		var context = this, args = arguments, now = +new Date;
		if( last && now < last + delay )
		{
		    clearTimeout( deferTimer );
		    deferTimer = setTimeout( function(){ last = now; fn.apply( context, args ); }, delay );
		}
		else
		{
		    last = now;
		    fn.apply( context, args );
		}
	    };
	};
    
    $window.on( 'scroll',
		throttle( throttleTimeout, function() {
		    dHeight			= $document.height();
		    wHeight			= $window.height();
		    wScrollCurrent	= $window.scrollTop();
		    wScrollDiff		= wScrollBefore - wScrollCurrent;
		    
		    if( wScrollCurrent <= 0 ) // scrolled to the very top; element sticks to the top
			$element.removeClass( elClassHidden );
		    
		    else if( wScrollDiff > 0 && $element.hasClass( elClassHidden ) ) // scrolled up; element slides in
			$element.removeClass( elClassHidden );
		    
		    else if( wScrollDiff < 0 ) // scrolled down
		    {
			if( wScrollCurrent + wHeight >= dHeight && $element.hasClass( elClassHidden ) ) // scrolled to the very bottom; element slides in
			    $element.removeClass( elClassHidden );
			
			else // scrolled down; element slides out
			    $element.addClass( elClassHidden );
		    }
		    
		    wScrollBefore = wScrollCurrent;
		}));
});


function openTab(evt, tabID) {
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
	tabcontent[i].className = tabcontent[i].className.replace(" active", "");
    }

    // Get all elements with class="tablink" and remove the class "active"
    tablinks = document.getElementsByClassName("tablink");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    //  as well as the content.
    contentDiv = document.getElementById(tabID)

    if( $(contentDiv).children().length == 0 ) {
	loadLocal('tabs/' + tabID + '.html', '#' + tabID, function() {
	    //do what we would have done below if it were loaded
	    contentDiv.style.display = "block";
	    contentDiv.className += " active";
	    render_katex( contentDiv );
	    enable_caption_toggles( contentDiv );
	    $(contentDiv).trigger('tabchange');
	    //enqueue_masonry( contentDiv ); //doesn't work yet
	});
    }
    else { //tab is already loaded
	contentDiv.style.display = "block";
	contentDiv.className += " active";

	// Run any onchange handlers on this tab in case switches have changed
	// (change handlers just exit if they're not in the active tab)
	$(contentDiv).trigger('tabchange');
    }
    evt.currentTarget.className += " active"; //do this right away always (no need to wait)
}

function render_katex(parentEl) {
    $("#status").show();
    $("#status").text("Rendering body math");
    $(parentEl).find(".math").each(function() {
        console.log("Rendering KateX");
        var texTxt = $(this).text();
        el = $(this).get(0);
        if(el.tagName == "DIV"){
            addDisp = "\\displaystyle";
        } else {
            addDisp = "";
        }
        try {
            katex.render(addDisp+texTxt, el);
        }
        catch(err) {
            $(this).html("<span class=\'err\'>"+err);
        }
    });
}

function enable_caption_toggles(parentEl) {
    $(parentEl).find("figcaption").each( function() { //old: .pygsti-wsoutput-group
	$(this).on('click', function() {
	    $(this).children("span.captiondetail").toggleClass('showcaption');}); //siblings("figcaption")
    });
}

function enqueue_masonry(parentEl) {
    plotman.enqueue( function() {

        $(parentEl).find('figure').each( function() {
            w = $(this).width();
            h = $(this).height();
            $(this).wrap( "<div class='griditem' style='" +
                          "width: " + w + "px; height: " + h + "px'></div>");
        });

        $(parentEl).masonry({
            itemSelector: '.griditem',
            stamp: 'h1',
            columnWidth: 100,
        });
    });
}


/* Set the width of the side navigation to open width and the left margin of the page content to the same */
function openNav() {
    document.getElementById("theSidenav").style.width = sidenav_width + "px";
    document.getElementById("main").style.marginLeft = sidenav_width + "px"; // for push: 
}

/* Set the width of the side navigation to 0 and the left margin of the page content to 0 */
function closeNav() {
    document.getElementById("theSidenav").style.width = sidenav_closed_width + "px";
    document.getElementById("main").style.marginLeft = sidenav_closed_width + "px"; // for push: 
}

function tackNav() {
    if(window.navtacked) {
	window.navtacked = false;
	document.getElementById("tackbtn").innerHTML = "&#8857;"
    } else {
	window.navtacked = true;
	document.getElementById("tackbtn").innerHTML = "&#8859;"
    }
}

function loadLocal(url, selector, complete) {
   var request = new XMLHttpRequest();
    request.responseType = 'text';
    request.open('GET', url, true);
    request.onload = function() {
      // "0" is usually an error, but Safari completes request and then sets
      // status to 0, probably b/c of cross-domain issues... use this as hack for now
      if (request.status == 200) { // || request.status == 0) {
          var response = request.responseText;
          $( selector ).html(response)
	  console.log("loadLocal: success loading " + request.responseURL);
	  complete();
      } else {
	  console.log("loadLocal: there was an error in the response:\n" + request.status);
      }
    };
    request.onerror = function() {
	console.log("loadLocal: there was a connection error of some sort!");
    };
    request.send();
}


function testLocalAjax(url, onerror) {
   var request = new XMLHttpRequest();
    request.responseType = 'text';
    request.open('GET', url, true);
    request.onload = function() {
	if (request.status != 200) {
	    onerror(request.status);
	}
	else {
	    console.log('testLocalAjax success!');
	}
    };
    request.onerror = function() {
	onerror("connection error");
    };
    request.send();
}

