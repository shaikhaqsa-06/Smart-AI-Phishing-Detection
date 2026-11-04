from flask import Flask, render_template, request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import re
import socket
import ipaddress
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tuned_random_forest_v3.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    print("=" * 60)
    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", type(model))
    print("Path:", MODEL_PATH)
    print("=" * 60)

except Exception as e:

    model = None

    print("=" * 60)
    print("MODEL LOAD ERROR")
    print(e)
    print("=" * 60)


# ============================================================
# FEATURE ORDER
# MUST MATCH TRAINING DATA EXACTLY
# ============================================================

FEATURE_NAMES = [

    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report"

]


# ============================================================
# REQUEST SETTINGS
# ============================================================

REQUEST_TIMEOUT = 8

HEADERS = {

    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"

}


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):

    if not url:
        return None

    url = url.strip()

    if not url:
        return None

    # Add HTTPS only when the user enters a normal domain.
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    try:

        parsed = urlparse(url)

        # Only HTTP and HTTPS are allowed.
        if parsed.scheme.lower() not in ["http", "https"]:
            return None

        # URL must contain a hostname.
        hostname = parsed.hostname

        if not hostname:
            return None

        hostname = hostname.lower().strip()

        # Hostname must contain a valid domain structure.
        # Reject simple words such as "hello".
        if "." not in hostname:
            return None

        # Reject invalid hostname characters.
        if not re.match(
            r"^[a-zA-Z0-9.-]+$",
            hostname
        ):
            return None

        # Hostname cannot start or end with a dot or hyphen.
        if (
            hostname.startswith(".")
            or hostname.endswith(".")
            or hostname.startswith("-")
            or hostname.endswith("-")
        ):
            return None

        # Reject consecutive dots.
        if ".." in hostname:
            return None

        return url

    except Exception:

        return None


# ============================================================
# PRIVATE / LOCAL IP PROTECTION
# ============================================================
# ============================================================
# PRIVATE / LOCAL IP PROTECTION
# ============================================================

def is_private_or_local_host(hostname):

    if not hostname:
        return True

    hostname = hostname.lower().strip()

    # Block local hostnames
    if hostname in [
        "localhost",
        "localhost.localdomain"
    ]:
        return True

    # Check if the hostname itself is an IP address
    try:

        ip = ipaddress.ip_address(hostname)

        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )

    except ValueError:

        pass

    # For normal domain names, resolve addresses and
    # block only addresses that are clearly private/local.
    try:

        addresses = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM
        )

        valid_public_address_found = False

        for address in addresses:

            ip_text = address[4][0]

            try:

                ip = ipaddress.ip_address(ip_text)

            except ValueError:

                continue

            # Clearly private/local address
            if (
                ip.is_loopback
                or ip.is_link_local
                or ip.is_private
                or ip.is_unspecified
                or ip.is_multicast
            ):

                continue

            # Public IP found
            valid_public_address_found = True

        # If at least one public IP exists, allow the domain.
        if valid_public_address_found:

            return False

    except Exception:

        # Do not block a normal domain only because
        # DNS information could not be completely resolved.
        return False

    return False

# ============================================================
# DOMAIN EXISTENCE CHECK
# ============================================================

def domain_exists(hostname):

    if not hostname:
        return False

    try:

        socket.gethostbyname(hostname)

        return True

    except socket.gaierror:

        return False

    except Exception:

        return False


# ============================================================
# FETCH WEBPAGE
# ============================================================

def fetch_page(url):

    try:

        parsed = urlparse(url)

        hostname = parsed.hostname

        if is_private_or_local_host(hostname):

            return {

                "success": False,
                "html": "",
                "final_url": url,
                "error":
                    "Private or local network addresses are not allowed."

            }


        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )


        final_url = response.url

        html = response.text or ""

        return {

            "success": True,
            "html": html,
            "final_url": final_url,
            "status_code": response.status_code,
            "redirect_count": len(response.history),
            "error": None

        }


    except requests.exceptions.RequestException as e:

        return {

            "success": False,
            "html": "",
            "final_url": url,
            "error": str(e)

        }


    except Exception as e:

        return {

            "success": False,
            "html": "",
            "final_url": url,
            "error": str(e)

        }


# ============================================================
# DOMAIN COMPARISON
# ============================================================

def get_hostname(url):

    try:

        return (urlparse(url).hostname or "").lower()

    except Exception:

        return ""


def same_domain(url1, url2):

    host1 = get_hostname(url1)
    host2 = get_hostname(url2)

    if not host1 or not host2:
        return False

    return (
        host1 == host2
        or host1.endswith("." + host2)
        or host2.endswith("." + host1)
    )


# ============================================================
# FEATURE: HAVING IP ADDRESS
# ============================================================

def feature_having_ip(url):

    hostname = get_hostname(url)

    try:

        ipaddress.ip_address(hostname)

        return -1

    except ValueError:

        return 1


# ============================================================
# FEATURE: URL LENGTH
# ============================================================

def feature_url_length(url):

    length = len(url)

    if length < 54:

        return 1

    elif length <= 75:

        return 0

    return -1


# ============================================================
# FEATURE: URL SHORTENING SERVICE
# ============================================================

def feature_shortening_service(url):

    hostname = get_hostname(url)

    shortening_services = [

        "bit.ly",
        "goo.gl",
        "tinyurl.com",
        "ow.ly",
        "t.co",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "bit.do",
        "cutt.ly",
        "shorturl.at",
        "rebrand.ly",
        "tiny.cc"

    ]

    for service in shortening_services:

        if hostname == service or hostname.endswith("." + service):

            return -1

    return 1


# ============================================================
# FEATURE: @ SYMBOL
# ============================================================

def feature_at_symbol(url):

    return -1 if "@" in url else 1


# ============================================================
# FEATURE: DOUBLE SLASH REDIRECTING
# ============================================================

def feature_double_slash(url):

    try:

        parsed = urlparse(url)

        path = parsed.path

        if "//" in path:

            return -1

        return 1

    except Exception:

        return 1


# ============================================================
# FEATURE: PREFIX / SUFFIX
# ============================================================

def feature_prefix_suffix(url):

    hostname = get_hostname(url)

    return -1 if "-" in hostname else 1


# ============================================================
# FEATURE: SUB DOMAIN
# ============================================================

def feature_sub_domain(url):

    hostname = get_hostname(url)

    if not hostname:

        return -1

    parts = hostname.split(".")

    if len(parts) <= 2:

        return 1

    elif len(parts) == 3:

        return 0

    return -1


# ============================================================
# FEATURE: SSL FINAL STATE
# ============================================================

def feature_ssl_state(url, page_data):

    parsed = urlparse(url)

    if parsed.scheme.lower() != "https":

        return -1

    if page_data.get("success"):

        return 1

    return 0


# ============================================================
# FEATURE: DOMAIN REGISTRATION LENGTH
# ============================================================

def feature_domain_registration_length(url):

    # WHOIS lookup is intentionally not performed.
    # 0 represents an unknown / intermediate state.

    return 0


# ============================================================
# FEATURE: FAVICON
# ============================================================

def feature_favicon(url, soup):

    if soup is None:

        return 0

    icons = soup.find_all(
        "link",
        href=True
    )

    for tag in icons:

        rel = tag.get("rel", [])

        rel_text = " ".join(rel).lower()

        if (
            "icon" in rel_text
            or "shortcut icon" in rel_text
        ):

            href = tag.get("href")

            if not href:

                continue

            icon_url = urljoin(
                url,
                href
            )

            if same_domain(
                url,
                icon_url
            ):

                return 1

            return -1

    return 0


# ============================================================
# FEATURE: PORT
# ============================================================

def feature_port(url):

    try:

        parsed = urlparse(url)

        port = parsed.port

        if port is None:

            return 1

        if (
            parsed.scheme == "http"
            and port == 80
        ):

            return 1

        if (
            parsed.scheme == "https"
            and port == 443
        ):

            return 1

        return -1

    except Exception:

        return 1


# ============================================================
# FEATURE: HTTPS TOKEN
# ============================================================

def feature_https_token(url):

    hostname = get_hostname(url)

    if "https" in hostname.lower():

        return -1

    return 1


# ============================================================
# EXTERNAL LINK RATIO
# ============================================================

def external_ratio(elements, base_url):

    if not elements:

        return 0

    total = 0
    external = 0

    for element in elements:

        href = (
            element.get("href")
            or element.get("src")
            or ""
        )

        if not href:

            continue

        total += 1

        full_url = urljoin(
            base_url,
            href
        )

        if not same_domain(
            base_url,
            full_url
        ):

            external += 1

    if total == 0:

        return 0

    return external / total


# ============================================================
# FEATURE: REQUEST URL
# ============================================================

def feature_request_url(url, soup):

    if soup is None:

        return 0

    elements = soup.find_all(
        ["img", "script", "link"]
    )

    ratio = external_ratio(
        elements,
        url
    )

    if ratio < 0.22:

        return 1

    elif ratio <= 0.61:

        return 0

    return -1


# ============================================================
# FEATURE: URL OF ANCHOR
# ============================================================

def feature_url_of_anchor(url, soup):

    if soup is None:

        return 0

    anchors = soup.find_all(
        "a",
        href=True
    )

    ratio = external_ratio(
        anchors,
        url
    )

    if ratio < 0.31:

        return 1

    elif ratio <= 0.67:

        return 0

    return -1


# ============================================================
# FEATURE: LINKS IN TAGS
# ============================================================

def feature_links_in_tags(url, soup):

    if soup is None:

        return 0

    elements = soup.find_all(
        ["meta", "script", "link"]
    )

    ratio = external_ratio(
        elements,
        url
    )

    if ratio < 0.17:

        return 1

    elif ratio <= 0.81:

        return 0

    return -1


# ============================================================
# FEATURE: SFH
# ============================================================

def feature_sfh(url, soup):

    if soup is None:

        return 0

    forms = soup.find_all(
        "form"
    )

    if not forms:

        return 1

    for form in forms:

        action = form.get(
            "action",
            ""
        ).strip()

        if (
            action == ""
            or action.lower() == "about:blank"
            or action.lower().startswith("javascript:")
        ):

            return -1

        action_url = urljoin(
            url,
            action
        )

        if same_domain(
            url,
            action_url
        ):

            continue

        return 0

    return 1


# ============================================================
# FEATURE: SUBMITTING TO EMAIL
# ============================================================

def feature_submitting_email(url, soup):

    if soup is None:

        return 1

    forms = soup.find_all(
        "form"
    )

    for form in forms:

        action = (
            form.get(
                "action",
                ""
            )
            .lower()
        )

        if (
            "mailto:" in action
            or "mail(" in action
            or "email" in action
        ):

            return -1

    page_text = soup.get_text(
        " ",
        strip=True
    ).lower()

    if "mailto:" in page_text:

        return -1

    return 1


# ============================================================
# FEATURE: ABNORMAL URL
# ============================================================

def feature_abnormal_url(url):

    hostname = get_hostname(url)

    if not hostname:

        return -1

    if (
        "@" in url
        or
        hostname.startswith("-")
        or
        hostname.endswith("-")
    ):

        return -1

    return 1


# ============================================================
# FEATURE: REDIRECT
# ============================================================

def feature_redirect(page_data):

    count = page_data.get(
        "redirect_count",
        0
    )

    if count == 0:

        return 1

    elif count <= 2:

        return 0

    return -1


# ============================================================
# FEATURE: ON MOUSEOVER
# ============================================================

def feature_mouseover(soup):

    if soup is None:

        return 1

    for tag in soup.find_all(
        onmouseover=True
    ):

        value = str(
            tag.get(
                "onmouseover",
                ""
            )
        ).lower()

        if (
            "window.status" in value
            or
            "status=" in value
        ):

            return -1

    return 1


# ============================================================
# FEATURE: RIGHT CLICK
# ============================================================

def feature_right_click(soup):

    if soup is None:

        return 1

    html = str(soup).lower()

    patterns = [

        "event.button==2",
        "event.button == 2",
        "contextmenu",
        "return false"

    ]

    for pattern in patterns:

        if pattern in html:

            return -1

    return 1


# ============================================================
# FEATURE: POPUP WINDOW
# ============================================================

def feature_popup_window(soup):

    if soup is None:

        return 1

    html = str(soup).lower()

    if (
        "window.open" in html
        or
        "alert(" in html
    ):

        return -1

    return 1


# ============================================================
# FEATURE: IFRAME
# ============================================================

def feature_iframe(soup):

    if soup is None:

        return 1

    if soup.find(
        "iframe"
    ) is not None:

        return -1

    if soup.find(
        "frame"
    ) is not None:

        return -1

    return 1


# ============================================================
# DOMAIN AGE
# ============================================================

def feature_age_of_domain(url):

    # No WHOIS service is used.
    # Unknown state is represented by 0.

    return 0


# ============================================================
# DNS RECORD
# ============================================================

def feature_dns_record(url):

    hostname = get_hostname(url)

    try:

        socket.gethostbyname(
            hostname
        )

        return 1

    except Exception:

        return -1


# ============================================================
# WEB TRAFFIC
# ============================================================

def feature_web_traffic(url):

    # External traffic ranking API is not used.
    # Unknown state = 0.

    return 0


# ============================================================
# PAGE RANK
# ============================================================

def feature_page_rank(url):

    # External PageRank service is not used.
    # Unknown state = 0.

    return 0


# ============================================================
# GOOGLE INDEX
# ============================================================

def feature_google_index(url):

    # A direct Google index check is not performed.
    # Unknown state = 0.

    return 0


# ============================================================
# LINKS POINTING TO PAGE
# ============================================================

def feature_links_pointing_to_page(url):

    # External backlink API is not used.
    # Unknown state = 0.

    return 0


# ============================================================
# STATISTICAL REPORT
# ============================================================

def feature_statistical_report(url):

    hostname = get_hostname(url)

    suspicious_words = [

        "login",
        "verify",
        "verification",
        "secure",
        "security",
        "account",
        "update",
        "confirm",
        "bank",
        "payment"

    ]

    count = 0

    for word in suspicious_words:

        if word in hostname:

            count += 1

    if count >= 2:

        return -1

    return 1


# ============================================================
# EXTRACT ALL 30 FEATURES
# ============================================================

def extract_features(url, page_data):

    html = page_data.get(
        "html",
        ""
    )

    try:

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

    except Exception:

        soup = None


    features = {}


    features[
        "having_IP_Address"
    ] = feature_having_ip(url)


    features[
        "URL_Length"
    ] = feature_url_length(url)


    features[
        "Shortining_Service"
    ] = feature_shortening_service(url)


    features[
        "having_At_Symbol"
    ] = feature_at_symbol(url)


    features[
        "double_slash_redirecting"
    ] = feature_double_slash(url)


    features[
        "Prefix_Suffix"
    ] = feature_prefix_suffix(url)


    features[
        "having_Sub_Domain"
    ] = feature_sub_domain(url)


    features[
        "SSLfinal_State"
    ] = feature_ssl_state(
        url,
        page_data
    )


    features[
        "Domain_registeration_length"
    ] = feature_domain_registration_length(
        url
    )


    features[
        "Favicon"
    ] = feature_favicon(
        url,
        soup
    )


    features[
        "port"
    ] = feature_port(url)


    features[
        "HTTPS_token"
    ] = feature_https_token(url)


    features[
        "Request_URL"
    ] = feature_request_url(
        url,
        soup
    )


    features[
        "URL_of_Anchor"
    ] = feature_url_of_anchor(
        url,
        soup
    )


    features[
        "Links_in_tags"
    ] = feature_links_in_tags(
        url,
        soup
    )


    features[
        "SFH"
    ] = feature_sfh(
        url,
        soup
    )


    features[
        "Submitting_to_email"
    ] = feature_submitting_email(
        url,
        soup
    )


    features[
        "Abnormal_URL"
    ] = feature_abnormal_url(url)


    features[
        "Redirect"
    ] = feature_redirect(
        page_data
    )


    features[
        "on_mouseover"
    ] = feature_mouseover(
        soup
    )


    features[
        "RightClick"
    ] = feature_right_click(
        soup
    )


    features[
        "popUpWidnow"
    ] = feature_popup_window(
        soup
    )


    features[
        "Iframe"
    ] = feature_iframe(
        soup
    )


    features[
        "age_of_domain"
    ] = feature_age_of_domain(
        url
    )


    features[
        "DNSRecord"
    ] = feature_dns_record(
        url
    )


    features[
        "web_traffic"
    ] = feature_web_traffic(
        url
    )


    features[
        "Page_Rank"
    ] = feature_page_rank(
        url
    )


    features[
        "Google_Index"
    ] = feature_google_index(
        url
    )


    features[
        "Links_pointing_to_page"
    ] = feature_links_pointing_to_page(
        url
    )


    features[
        "Statistical_report"
    ] = feature_statistical_report(
        url
    )


    return features


# ============================================================
# FEATURE STATUS FOR UI
# ============================================================

def build_feature_check(features):

    feature_check = []


    for feature_name in FEATURE_NAMES:

        value = features.get(
            feature_name,
            0
        )


        if value == -1:

            status = "Suspicious"

        elif value == 1:

            status = "Normal"

        else:

            status = "Unknown / Neutral"


        feature_check.append({

            "name": feature_name,
            "value": value,
            "status": status

        })


    return feature_check


# ============================================================
# RISK INDICATORS
# ============================================================

def build_risk_indicators(
    url,
    features
):

    risks = []


    if features.get(
        "having_IP_Address"
    ) == -1:

        risks.append(
            "The URL uses an IP address instead of a normal domain name."
        )


    if features.get(
        "URL_Length"
    ) == -1:

        risks.append(
            "The URL is unusually long."
        )


    if features.get(
        "Shortining_Service"
    ) == -1:

        risks.append(
            "A URL shortening service was detected."
        )


    if features.get(
        "having_At_Symbol"
    ) == -1:

        risks.append(
            "The URL contains an @ symbol."
        )


    if features.get(
        "double_slash_redirecting"
    ) == -1:

        risks.append(
            "Suspicious double-slash redirection was detected."
        )


    if features.get(
        "Prefix_Suffix"
    ) == -1:

        risks.append(
            "A hyphen was detected in the domain name."
        )


    if features.get(
        "having_Sub_Domain"
    ) == -1:

        risks.append(
            "Multiple sub-domains were detected."
        )


    if features.get(
        "SSLfinal_State"
    ) == -1:

        risks.append(
            "The URL does not use HTTPS."
        )


    if features.get(
        "HTTPS_token"
    ) == -1:

        risks.append(
            "The domain contains the HTTPS token suspiciously."
        )


    if features.get(
        "Submitting_to_email"
    ) == -1:

        risks.append(
            "The webpage appears to submit information through email."
        )


    if features.get(
        "Abnormal_URL"
    ) == -1:

        risks.append(
            "The URL contains abnormal characteristics."
        )


    if features.get(
        "Redirect"
    ) == -1:

        risks.append(
            "Multiple redirects were detected."
        )


    if features.get(
        "on_mouseover"
    ) == -1:

        risks.append(
            "Suspicious mouse-over behavior was detected."
        )


    if features.get(
        "RightClick"
    ) == -1:

        risks.append(
            "Right-click disabling behavior was detected."
        )


    if features.get(
        "popUpWidnow"
    ) == -1:

        risks.append(
            "Suspicious popup behavior was detected."
        )


    if features.get(
        "Iframe"
    ) == -1:

        risks.append(
            "An iframe/frame was detected."
        )


    if features.get(
        "Statistical_report"
    ) == -1:

        risks.append(
            "The domain contains suspicious security-related terms."
        )


    return risks


# ============================================================
# PREDICTION
# ============================================================

def predict_url(features):

    if model is None:

        raise RuntimeError(
            "Machine learning model could not be loaded."
        )


    values = [

        features.get(
            feature_name,
            0
        )

        for feature_name in FEATURE_NAMES

    ]


    input_data = pd.DataFrame(
        [values],
        columns=FEATURE_NAMES
    )


    prediction = model.predict(
        input_data
    )[0]


    result = (
        "PHISHING"
        if prediction == -1
        else "LEGITIMATE"
    )


    confidence = 0.0


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            input_data
        )[0]


        classes = list(
            model.classes_
        )


        try:

            prediction_index = classes.index(
                prediction
            )

            confidence = (
                float(
                    probabilities[
                        prediction_index
                    ]
                )
                * 100
            )

        except Exception:

            confidence = (
                float(
                    np.max(
                        probabilities
                    )
                )
                * 100
            )


    return result, confidence


# ============================================================
# MAIN ROUTE
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)

def index():

    scanned_url = ""

    confidence = 0

    result = None

    error_message = None

    final_url = None

    risk_indicators = []

    feature_check = []


    if request.method == "POST":

        scanned_url = (
            request.form.get(
                "url",
                ""
            )
            .strip()
        )


        # ====================================================
        # STEP 1: VALIDATE URL
        # ====================================================

        normalized_url = normalize_url(
            scanned_url
        )


        if not normalized_url:

            error_message = (
                "Please enter a valid HTTP or HTTPS URL."
            )

            return render_template(
                "index.html",
                scanned_url=scanned_url,
                confidence=0,
                result=None,
                error_message=error_message,
                final_url=None,
                risk_indicators=[],
                feature_check=[]
            )


        scanned_url = normalized_url


        hostname = get_hostname(
            normalized_url
        )


        # ====================================================
        # STEP 2: BLOCK PRIVATE / LOCAL HOSTS
        # ====================================================

        if is_private_or_local_host(
            hostname
        ):

            error_message = (
                "Private or local network URLs are not allowed."
            )

            return render_template(
                "index.html",
                scanned_url=scanned_url,
                confidence=0,
                result=None,
                error_message=error_message,
                final_url=None,
                risk_indicators=[],
                feature_check=[]
            )


        # ====================================================
        # STEP 3: CHECK DOMAIN EXISTENCE
        #
        # If the domain does not exist, do NOT allow the
        # machine learning model to call it legitimate.
        # ====================================================

        if not domain_exists(hostname):

            result = "PHISHING"

            confidence = 100

            error_message = (
                "The domain does not exist or could not be "
                "verified through DNS. The URL is treated "
                "as PHISHING / FAKE."
            )

            risk_indicators = [

                "The domain could not be found or verified through DNS."

            ]

            return render_template(
                "index.html",
                scanned_url=scanned_url,
                confidence=confidence,
                result=result,
                error_message=error_message,
                final_url=None,
                risk_indicators=risk_indicators,
                feature_check=[]
            )


        # ====================================================
        # STEP 4: FETCH WEBSITE + ML ANALYSIS
        # ====================================================

        try:

            page_data = fetch_page(
                normalized_url
            )


            final_url = page_data.get(
                "final_url",
                normalized_url
            )


            features = extract_features(
                normalized_url,
                page_data
            )


            feature_check = build_feature_check(
                features
            )


            risk_indicators = build_risk_indicators(
                normalized_url,
                features
            )


            result, confidence = predict_url(
                features
            )


            confidence = round(
                float(confidence),
                2
            )


        except Exception as e:

            error_message = (
                "An error occurred while scanning the URL."
            )

            print(
                "SCAN ERROR:",
                repr(e)
            )


    return render_template(

        "index.html",

        scanned_url=scanned_url,

        confidence=confidence,

        result=result,

        error_message=error_message,

        final_url=final_url,

        risk_indicators=risk_indicators,

        feature_check=feature_check

    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SMART AI PHISHING URL BLOCKER")
    print("=" * 60)
    print("Starting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )