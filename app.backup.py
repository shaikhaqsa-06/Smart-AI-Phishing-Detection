from flask import Flask, render_template, request
from pathlib import Path
from urllib.parse import urlparse, urljoin
import socket
import ipaddress
import re
import warnings

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

app = Flask(__name__)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tuned_random_forest_v3.pkl"

# =========================================================
# LOAD MODEL
# =========================================================

MODEL = None
MODEL_ERROR = None

try:
    MODEL = joblib.load(MODEL_PATH)
    print("==============================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", MODEL_PATH)
    print("Model type:", type(MODEL))
    print("==============================================")
except Exception as e:
    MODEL_ERROR = str(e)
    print("==============================================")
    print("MODEL LOAD ERROR")
    print(MODEL_ERROR)
    print("==============================================")


# =========================================================
# EXACT 30 FEATURES USED BY TRAINED MODEL
# =========================================================

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


# =========================================================
# URL NORMALIZATION
# =========================================================

def normalize_url(url):
    if not url:
        return None

    url = url.strip()

    if not url:
        return None

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    try:
        parsed = urlparse(url)

        if parsed.scheme.lower() not in ("http", "https"):
            return None

        if not parsed.hostname:
            return None

        return url

    except Exception:
        return None


# =========================================================
# IP / HOST SECURITY
# =========================================================

def resolve_hostname(hostname):
    addresses = set()

    try:
        results = socket.getaddrinfo(
            hostname,
            None,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM
        )

        for result in results:
            try:
                ip = result[4][0]
                addresses.add(ip)
            except Exception:
                pass

    except Exception:
        pass

    return list(addresses)


def is_public_ip(ip):
    try:
        obj = ipaddress.ip_address(ip)

        if obj.is_private:
            return False

        if obj.is_loopback:
            return False

        if obj.is_link_local:
            return False

        if obj.is_multicast:
            return False

        if obj.is_reserved:
            return False

        if obj.is_unspecified:
            return False

        return obj.is_global

    except Exception:
        return False


def is_public_hostname(hostname):

    if not hostname:
        return False

    hostname = hostname.lower().strip()

    blocked_hosts = {
        "localhost",
        "localhost.localdomain"
    }

    if hostname in blocked_hosts:
        return False

    # Direct IP
    try:
        ip_obj = ipaddress.ip_address(hostname)

        return is_public_ip(str(ip_obj))

    except ValueError:
        pass

    addresses = resolve_hostname(hostname)

    # IMPORTANT:
    # We do NOT reject the host just because one address
    # is unavailable/non-public.
    #
    # At least one globally reachable address is enough.

    for address in addresses:
        if is_public_ip(address):
            return True

    return False


# =========================================================
# SAFE WEBSITE FETCH
# =========================================================

def fetch_page(url):

    parsed = urlparse(url)
    hostname = parsed.hostname

    # -----------------------------------------------------
    # Public website check
    # -----------------------------------------------------

    if not is_public_hostname(hostname):
        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": "This URL could not be verified as a public website."
        }

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=(5, 8),
            allow_redirects=True,
            verify=True,
            stream=True
        )

        final_url = response.url

        # -------------------------------------------------
        # Check redirect destination
        # -------------------------------------------------

        final_host = urlparse(final_url).hostname

        if final_host and not is_public_hostname(final_host):
            return {
                "success": False,
                "html": "",
                "final_url": final_url,
                "error": (
                    "The website redirected to a private or "
                    "non-public network address."
                )
            }

        # -------------------------------------------------
        # Read limited HTML
        # -------------------------------------------------

        content = b""

        max_bytes = 1024 * 1024  # 1 MB

        for chunk in response.iter_content(chunk_size=8192):

            if not chunk:
                continue

            content += chunk

            if len(content) >= max_bytes:
                break

        try:
            html = content.decode(
                response.encoding or "utf-8",
                errors="ignore"
            )
        except Exception:
            html = content.decode(
                "utf-8",
                errors="ignore"
            )

        return {
            "success": True,
            "html": html,
            "final_url": final_url,
            "error": None
        }

    except requests.exceptions.SSLError:

        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": "The website has an SSL certificate problem."
        }

    except requests.exceptions.Timeout:

        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": "The website took too long to respond."
        }

    except requests.exceptions.ConnectionError:

        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": "The website could not be reached."
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": f"Website request failed: {str(e)}"
        }

    except Exception as e:

        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": f"Unexpected website error: {str(e)}"
        }


# =========================================================
# URL FEATURES
# =========================================================

SHORTENING_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "cutt.ly",
    "shorturl.at",
    "rebrand.ly",
    "tiny.cc",
    "lnkd.in"
}


def is_shortening_service(hostname):

    if not hostname:
        return False

    hostname = hostname.lower().strip()

    hostname = hostname.split(":")[0]

    for domain in SHORTENING_DOMAINS:

        if hostname == domain:
            return True

        if hostname.endswith("." + domain):
            return True

    return False


def extract_url_features(original_url):

    features = {}

    parsed = urlparse(original_url)

    hostname = parsed.hostname or ""

    hostname_lower = hostname.lower()

    # -----------------------------------------------------
    # 1. having_IP_Address
    # -----------------------------------------------------

    try:
        ipaddress.ip_address(hostname)
        features["having_IP_Address"] = -1
    except Exception:
        features["having_IP_Address"] = 1

    # -----------------------------------------------------
    # 2. URL_Length
    # -----------------------------------------------------

    length = len(original_url)

    if length < 54:
        features["URL_Length"] = 1

    elif length <= 75:
        features["URL_Length"] = 0

    else:
        features["URL_Length"] = -1

    # -----------------------------------------------------
    # 3. Shortining_Service
    # -----------------------------------------------------

    if is_shortening_service(hostname_lower):
        features["Shortining_Service"] = 1
    else:
        features["Shortining_Service"] = -1

    # -----------------------------------------------------
    # 4. having_At_Symbol
    # -----------------------------------------------------

    features["having_At_Symbol"] = (
        1 if "@" in original_url else -1
    )

    # -----------------------------------------------------
    # 5. double_slash_redirecting
    # -----------------------------------------------------

    after_protocol = re.sub(
        r"^https?://",
        "",
        original_url,
        flags=re.IGNORECASE
    )

    if "//" in after_protocol:
        features["double_slash_redirecting"] = -1
    else:
        features["double_slash_redirecting"] = 1

    # -----------------------------------------------------
    # 6. Prefix_Suffix
    # -----------------------------------------------------

    if "-" in hostname_lower:
        features["Prefix_Suffix"] = -1
    else:
        features["Prefix_Suffix"] = 1

    # -----------------------------------------------------
    # 7. having_Sub_Domain
    # -----------------------------------------------------

    host_without_www = hostname_lower

    if host_without_www.startswith("www."):
        host_without_www = host_without_www[4:]

    parts = host_without_www.split(".")

    if len(parts) <= 2:
        features["having_Sub_Domain"] = 1

    elif len(parts) == 3:
        features["having_Sub_Domain"] = 0

    else:
        features["having_Sub_Domain"] = -1

    # -----------------------------------------------------
    # 8. SSLfinal_State
    # -----------------------------------------------------

    if parsed.scheme.lower() == "https":
        features["SSLfinal_State"] = 1
    else:
        features["SSLfinal_State"] = -1

    # -----------------------------------------------------
    # 9. Domain_registeration_length
    #
    # External WHOIS lookup is intentionally not performed.
    # Neutral value is used.
    # -----------------------------------------------------

    features["Domain_registeration_length"] = 0

    # -----------------------------------------------------
    # 10. Favicon
    # -----------------------------------------------------

    features["Favicon"] = 0

    # -----------------------------------------------------
    # 11. port
    # -----------------------------------------------------

    try:
        port = parsed.port

        if port is None or port in (80, 443):
            features["port"] = 1
        else:
            features["port"] = -1

    except Exception:
        features["port"] = -1

    # -----------------------------------------------------
    # 12. HTTPS_token
    # -----------------------------------------------------

    if "https" in hostname_lower:
        features["HTTPS_token"] = 1
    else:
        features["HTTPS_token"] = -1

    # -----------------------------------------------------
    # HTML dependent features start neutral
    # -----------------------------------------------------

    features["Request_URL"] = 0
    features["URL_of_Anchor"] = 0
    features["Links_in_tags"] = 0
    features["SFH"] = 0

    # -----------------------------------------------------
    # 17. Submitting_to_email
    # -----------------------------------------------------

    features["Submitting_to_email"] = -1

    # -----------------------------------------------------
    # 18. Abnormal_URL
    #
    # Initial value is normal.
    # Redirect analysis may change it.
    # -----------------------------------------------------

    features["Abnormal_URL"] = -1

    # -----------------------------------------------------
    # 19. Redirect
    #
    # UCI representation uses 0/1.
    # -----------------------------------------------------

    features["Redirect"] = 0

    # -----------------------------------------------------
    # Browser behavior features
    # -----------------------------------------------------

    features["on_mouseover"] = -1
    features["RightClick"] = -1
    features["popUpWidnow"] = -1
    features["Iframe"] = -1

    # -----------------------------------------------------
    # External/domain features
    #
    # Neutral because no external reputation service is used.
    # -----------------------------------------------------

    features["age_of_domain"] = 0
    features["DNSRecord"] = 0
    features["web_traffic"] = 0
    features["Page_Rank"] = 0
    features["Google_Index"] = 0
    features["Links_pointing_to_page"] = 0
    features["Statistical_report"] = 0

    return features


# =========================================================
# PAGE FEATURES
# =========================================================

def same_domain(url1, url2):

    try:

        host1 = (urlparse(url1).hostname or "").lower()
        host2 = (urlparse(url2).hostname or "").lower()

        if host1.startswith("www."):
            host1 = host1[4:]

        if host2.startswith("www."):
            host2 = host2[4:]

        return host1 == host2

    except Exception:
        return False


def external_ratio(urls, base_url):

    if not urls:
        return 0.0

    total = len(urls)
    external = 0

    for item in urls:

        try:
            absolute = urljoin(base_url, item)

            if not same_domain(base_url, absolute):
                external += 1

        except Exception:
            external += 1

    return external / total


def extract_page_features(html, base_url):

    features = {}

    if not html:
        return features

    try:

        soup = BeautifulSoup(html, "html.parser")

        # -------------------------------------------------
        # FAVICON
        # -------------------------------------------------

        favicon_found = False

        for link in soup.find_all("link"):

            rel = link.get("rel", [])

            if isinstance(rel, list):
                rel_text = " ".join(rel).lower()
            else:
                rel_text = str(rel).lower()

            if "icon" in rel_text:
                favicon_found = True
                break

        features["Favicon"] = 1 if favicon_found else -1

        # -------------------------------------------------
        # ANCHORS
        # -------------------------------------------------

        anchors = []

        for tag in soup.find_all("a", href=True):
            anchors.append(tag.get("href"))

        ratio = external_ratio(anchors, base_url)

        if ratio > 0.67:
            features["URL_of_Anchor"] = -1

        elif ratio > 0.31:
            features["URL_of_Anchor"] = 0

        else:
            features["URL_of_Anchor"] = 1

        # -------------------------------------------------
        # LINKS IN TAGS
        # -------------------------------------------------

        link_values = []

        for tag in soup.find_all(
            ["link", "script", "img"],
            src=True
        ):
            link_values.append(tag.get("src"))

        for tag in soup.find_all(
            ["link"],
            href=True
        ):
            link_values.append(tag.get("href"))

        ratio = external_ratio(
            link_values,
            base_url
        )

        if ratio > 0.81:
            features["Links_in_tags"] = -1

        elif ratio > 0.17:
            features["Links_in_tags"] = 0

        else:
            features["Links_in_tags"] = 1

        # -------------------------------------------------
        # REQUEST_URL
        # -------------------------------------------------

        media_values = []

        for tag in soup.find_all(
            ["img", "audio", "video", "embed", "iframe"],
            src=True
        ):
            media_values.append(tag.get("src"))

        ratio = external_ratio(
            media_values,
            base_url
        )

        if ratio > 0.61:
            features["Request_URL"] = -1

        elif ratio > 0.22:
            features["Request_URL"] = 0

        else:
            features["Request_URL"] = 1

        # -------------------------------------------------
        # FORMS / SFH
        # -------------------------------------------------

        suspicious_form = False

        for form in soup.find_all("form"):

            action = (
                form.get("action") or ""
            ).strip().lower()

            if (
                action == ""
                or action.startswith("javascript:")
                or action.startswith("mailto:")
                or action.startswith("about:blank")
            ):
                suspicious_form = True
                break

            absolute_action = urljoin(
                base_url,
                action
            )

            if not same_domain(
                base_url,
                absolute_action
            ):
                suspicious_form = True
                break

        features["SFH"] = (
            -1 if suspicious_form else 1
        )

        # -------------------------------------------------
        # SUBMITTING TO EMAIL
        # -------------------------------------------------

        email_found = False

        page_text = html.lower()

        if "mailto:" in page_text:
            email_found = True

        if "mail(" in page_text:
            email_found = True

        if "sendmail" in page_text:
            email_found = True

        features["Submitting_to_email"] = (
            1 if email_found else -1
        )

        # -------------------------------------------------
        # MOUSEOVER
        # -------------------------------------------------

        features["on_mouseover"] = (
            1
            if re.search(
                r"\bonmouseover\s*=",
                html,
                re.IGNORECASE
            )
            else -1
        )

        # -------------------------------------------------
        # RIGHT CLICK
        # -------------------------------------------------

        right_click_patterns = [
            r"oncontextmenu\s*=",
            r"event\s*\.\s*button\s*==\s*2",
            r"\bcontextmenu\b"
        ]

        right_click_found = any(
            re.search(
                pattern,
                html,
                re.IGNORECASE
            )
            for pattern in right_click_patterns
        )

        features["RightClick"] = (
            1 if right_click_found else -1
        )

        # -------------------------------------------------
        # POPUP
        # -------------------------------------------------

        popup_patterns = [
            r"window\s*\.\s*open\s*\(",
            r"\balert\s*\("
        ]

        popup_found = any(
            re.search(
                pattern,
                html,
                re.IGNORECASE
            )
            for pattern in popup_patterns
        )

        features["popUpWidnow"] = (
            1 if popup_found else -1
        )

        # -------------------------------------------------
        # IFRAME
        # -------------------------------------------------

        iframe_found = bool(
            soup.find("iframe")
        )

        features["Iframe"] = (
            1 if iframe_found else -1
        )

        return features

    except Exception:

        return features


# =========================================================
# COMBINE ALL FEATURES
# =========================================================

def build_features(
    original_url,
    final_url,
    html
):

    features = {}

    # URL features
    url_features = extract_url_features(
        original_url
    )

    features.update(url_features)

    # Page features
    if html:

        page_features = extract_page_features(
            html,
            final_url
        )

        features.update(page_features)

    # -----------------------------------------------------
    # REDIRECT
    # -----------------------------------------------------

    if final_url.rstrip("/") != original_url.rstrip("/"):
        features["Redirect"] = 1
    else:
        features["Redirect"] = 0

    # -----------------------------------------------------
    # ABNORMAL URL
    #
    # Only mark abnormal when redirect ends on another host.
    # -----------------------------------------------------

    try:

        original_host = (
            urlparse(original_url).hostname or ""
        ).lower()

        final_host = (
            urlparse(final_url).hostname or ""
        ).lower()

        if original_host.startswith("www."):
            original_host = original_host[4:]

        if final_host.startswith("www."):
            final_host = final_host[4:]

        if original_host != final_host:
            features["Abnormal_URL"] = 1
        else:
            features["Abnormal_URL"] = -1

    except Exception:

        features["Abnormal_URL"] = -1

    # -----------------------------------------------------
    # Make sure every feature exists
    # -----------------------------------------------------

    for name in FEATURE_NAMES:

        if name not in features:
            features[name] = 0

    # Exact order
    features = {
        name: features[name]
        for name in FEATURE_NAMES
    }

    return features


# =========================================================
# FEATURE EXPLANATIONS
# =========================================================

FEATURE_EXPLANATIONS = {

    "having_IP_Address":
        "IP address usage in the URL.",

    "URL_Length":
        "Length of the URL.",

    "Shortining_Service":
        "Checks whether a known URL shortening service is used.",

    "having_At_Symbol":
        "Checks for the @ symbol in the URL.",

    "double_slash_redirecting":
        "Checks for suspicious double-slash redirection patterns.",

    "Prefix_Suffix":
        "Checks whether the domain contains a hyphen.",

    "having_Sub_Domain":
        "Checks the number of subdomains.",

    "SSLfinal_State":
        "Checks whether HTTPS is used.",

    "Domain_registeration_length":
        "Domain registration duration information.",

    "Favicon":
        "Checks for a webpage favicon.",

    "port":
        "Checks whether a standard web port is used.",

    "HTTPS_token":
        "Checks for the HTTPS token inside the hostname.",

    "Request_URL":
        "Analyzes external resources requested by the webpage.",

    "URL_of_Anchor":
        "Analyzes external links in webpage anchors.",

    "Links_in_tags":
        "Analyzes external links inside HTML tags.",

    "SFH":
        "Analyzes form submission destinations.",

    "Submitting_to_email":
        "Checks whether forms or scripts submit data to email.",

    "Abnormal_URL":
        "Checks abnormal URL or cross-domain redirection behavior.",

    "Redirect":
        "Checks whether the URL redirects.",

    "on_mouseover":
        "Checks for mouseover-based browser behavior.",

    "RightClick":
        "Checks for right-click disabling behavior.",

    "popUpWidnow":
        "Checks for popup or alert behavior.",

    "Iframe":
        "Checks whether iframe elements are present.",

    "age_of_domain":
        "Domain age information.",

    "DNSRecord":
        "DNS record availability information.",

    "web_traffic":
        "Website traffic information.",

    "Page_Rank":
        "Page ranking information.",

    "Google_Index":
        "Search engine indexing information.",

    "Links_pointing_to_page":
        "Number/category of links pointing to the page.",

    "Statistical_report":
        "External statistical/reputation information."
}


# =========================================================
# RISK INDICATORS
# =========================================================

def get_risk_indicators(features):

    risks = []

    # -----------------------------------------------------
    # URL based risks
    # -----------------------------------------------------

    if features.get("having_IP_Address") == -1:
        risks.append(
            "The URL uses an IP address instead of a domain name."
        )

    if features.get("URL_Length") == -1:
        risks.append(
            "The URL is unusually long."
        )

    if features.get("Shortining_Service") == 1:
        risks.append(
            "A URL shortening service is being used."
        )

    if features.get("having_At_Symbol") == 1:
        risks.append(
            "The URL contains an @ symbol."
        )

    if features.get("double_slash_redirecting") == -1:
        risks.append(
            "Suspicious double-slash redirection pattern detected."
        )

    if features.get("Prefix_Suffix") == -1:
        risks.append(
            "The domain contains a hyphen."
        )

    if features.get("having_Sub_Domain") == -1:
        risks.append(
            "The URL contains multiple subdomains."
        )

    if features.get("SSLfinal_State") == -1:
        risks.append(
            "The URL does not use HTTPS."
        )

    if features.get("port") == -1:
        risks.append(
            "A non-standard port is being used."
        )

    if features.get("Redirect") == 1:
        risks.append(
            "The requested URL was redirected."
        )

    if features.get("Abnormal_URL") == 1:
        risks.append(
            "The URL redirected to a different host."
        )

    # -----------------------------------------------------
    # Page based risks
    # -----------------------------------------------------

    if features.get("URL_of_Anchor") == -1:
        risks.append(
            "Many webpage links point to external domains."
        )

    if features.get("Links_in_tags") == -1:
        risks.append(
            "Many webpage resources are loaded from external domains."
        )

    if features.get("Request_URL") == -1:
        risks.append(
            "The webpage loads many external resources."
        )

    if features.get("SFH") == -1:
        risks.append(
            "The webpage contains a suspicious form submission target."
        )

    if features.get("Submitting_to_email") == 1:
        risks.append(
            "The webpage appears to submit information through email."
        )

    if features.get("on_mouseover") == 1:
        risks.append(
            "Mouseover-based scripting was detected."
        )

    if features.get("RightClick") == 1:
        risks.append(
            "Right-click disabling behavior was detected."
        )

    if features.get("popUpWidnow") == 1:
        risks.append(
            "Popup or alert behavior was detected."
        )

    if features.get("Iframe") == 1:
        risks.append(
            "The webpage contains iframe elements."
        )

    return risks


# =========================================================
# FEATURE CHECK
# =========================================================

def make_feature_check(features):

    result = []

    for name in FEATURE_NAMES:

        value = features.get(name, 0)

        explanation = FEATURE_EXPLANATIONS.get(
            name,
            "Feature analyzed by the machine learning model."
        )

        if value == 1:
            status = "Positive / Detected"

        elif value == -1:
            status = "Negative / Not detected"

        else:
            status = "Unknown / Neutral"

        result.append({
            "name": name,
            "value": value,
            "status": status,
            "description": explanation
        })

    return result


# =========================================================
# MACHINE LEARNING PREDICTION
# =========================================================

def predict_url(features):

    if MODEL is None:
        raise RuntimeError(
            f"Model could not be loaded: {MODEL_ERROR}"
        )

    # -----------------------------------------------------
    # EXACT FEATURE ORDER
    # -----------------------------------------------------

    X = pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = MODEL.predict(X)[0]

    # UCI dataset:
    # -1 = Phishing
    #  1 = Legitimate

    if prediction == -1:
        result = "PHISHING"
    else:
        result = "LEGITIMATE"

    # -----------------------------------------------------
    # Probability
    # -----------------------------------------------------

    confidence = None

    try:

        probabilities = MODEL.predict_proba(X)[0]

        classes = list(MODEL.classes_)

        if prediction in classes:

            index = classes.index(prediction)

            confidence = float(
                probabilities[index] * 100
            )

    except Exception:
        confidence = None

    return result, confidence


# =========================================================
# MAIN ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    scanned_url = None
    final_url = None
    risk_indicators = []
    feature_check = []
    error_message = None
    scan_status = None

    if request.method == "POST":

        user_url = request.form.get(
            "url",
            ""
        ).strip()

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        original_url = normalize_url(
            user_url
        )

        if not original_url:

            error_message = (
                "Please enter a valid HTTP or HTTPS URL."
            )

            return render_template(
                "index.html",
                result=result,
                confidence=confidence,
                scanned_url=scanned_url,
                final_url=final_url,
                risk_indicators=risk_indicators,
                feature_check=feature_check,
                error_message=error_message,
                scan_status=scan_status
            )

        scanned_url = original_url

        # -------------------------------------------------
        # Fetch website
        # -------------------------------------------------

        page_result = fetch_page(
            original_url
        )

        html = page_result["html"]

        final_url = page_result["final_url"]

        # -------------------------------------------------
        # IMPORTANT:
        #
        # Even if website cannot be fetched,
        # prediction continues using URL features.
        # -------------------------------------------------

        scan_status = (
            "Webpage analyzed successfully."
            if page_result["success"]
            else
            "Webpage could not be fetched. "
            "URL-based features were used for ML prediction."
        )

        # -------------------------------------------------
        # Build 30 features
        # -------------------------------------------------

        features = build_features(
            original_url,
            final_url,
            html
        )

        # -------------------------------------------------
        # ML prediction
        # -------------------------------------------------

        try:

            result, confidence = predict_url(
                features
            )

            risk_indicators = get_risk_indicators(
                features
            )

            feature_check = make_feature_check(
                features
            )

        except Exception as e:

            error_message = (
                f"Prediction error: {str(e)}"
            )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        scanned_url=scanned_url,
        final_url=final_url,
        risk_indicators=risk_indicators,
        feature_check=feature_check,
        error_message=error_message,
        scan_status=scan_status
    )


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    print("==============================================")
    print("SMART AI PHISHING URL BLOCKER")
    print("==============================================")
    print("Model path:", MODEL_PATH)
    print("Model exists:", MODEL_PATH.exists())

    if MODEL_PATH.exists():
        print(
            "Model size:",
            MODEL_PATH.stat().st_size,
            "bytes"
        )

    print("Server: http://127.0.0.1:5000")
    print("==============================================")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )