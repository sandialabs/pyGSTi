function render_katex(parentEl) {
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


function openTab(tabID) {
    var targetTab = $(`#${tabID}`);  // hacky
    
    if(!targetTab.is(":visible")) {
        // Fade out open tab
        $('.tabcontent').hide().removeClass('active');
        // Remove active style on any tab links
        $('.tablink').removeClass('active');

        //load the tab content on first use
        load_contents_if_necessary(tabID);
        
        // Fade in target tab
        targetTab.show().addClass('active').trigger('tabchange');
        // Add active style to event target link, presumably clicked to open this tab
        //$(evt.target).addClass('active');  //(this doesn't work as evt is undefined)
    }
}


function load_contents_if_necessary(itemID) {
    var target = $(`#${itemID}`);  // hacky
    if(target.hasClass('notloaded')) {
        target.trigger('load_loadable_item');
        target.removeClass('notloaded');
    }
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

function initNav() {
    load_contents_if_necessary('theSidenav')  // always active
    window.navtacked = true;
    openNav();

    //Mousemove handler
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
}

function initBanner() {
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
}


sidenav_width = 200;
sidenav_closed_width = 10;
sidenav_mousetrigger_width = 50;

$(document).ready(function() {
    // Create window plot manager
    window.plotman = new PlotManager();

    // Render KaTeX
    render_katex('body');

    // Iterate through all figure captions and add a default caption detail
    const figcaptions = document.getElementsByTagName("figcaption")
    for (const figcap of figcaptions) {
        const defaultcaption = document.createElement('span')
        defaultcaption.className = 'defaultcaptiondetail'
        defaultcaption.innerHTML = '(Click to expand details)'
        defaultcaption.classList.toggle("showcaption")
        figcap.appendChild(defaultcaption)
    }

    // Enable figure caption toggling
    $('figcaption').on('click', function() {
        // captiondetails should be divs, not spans
        $(this).children('.captiondetail').toggleClass('showcaption')
        // Also turn off default caption
        $(this).children('.defaultcaptiondetail').toggleClass('showcaption')
    });
});


function testLocalAjax(url, onerror) {
   var request = new XMLHttpRequest();
    request.responseType = 'text';
    request.withCredentials = true; //b/c jupyter notebooks use user authentication
    request.open('GET', url, true);
    request.onload = function() {
	var is_safari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
	var is_chrome = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
	if (request.status == 200 || ((is_safari || is_chrome) && request.status == 0)) {
	    console.log('testLocalAjax success!');
	}
	else {
	    onerror(request.status);
	}
    };
    request.onerror = function() {
	onerror("connection error");
    };
    request.send();
}
