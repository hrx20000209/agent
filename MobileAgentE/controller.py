import os
import re
import time
import subprocess
from PIL import Image
from time import sleep


_PATTERN_TO_ACTIVITY = {
    r'google chrome|chrome':
        'com.android.chrome/com.google.android.apps.chrome.Main',

    r'google chat':
        'com.google.android.apps.dynamite/com.google.android.apps.dynamite.startup.StartUpActivity',

    r'settings|system settings':
        'com.android.settings/.Settings',

    r'youtube|yt':
        'com.google.android.youtube/com.google.android.apps.youtube.app.WatchWhileActivity',

    r'google play|play store|gps':
        'com.android.vending/com.google.android.finsky.activities.MainActivity',

    r'gmail|gemail|google mail|google email|google mail client':
        'com.google.android.gm/.ConversationListActivityGmail',

    r'google maps|gmaps|maps|google map':
        'com.google.android.apps.maps/com.google.android.maps.MapsActivity',

    r'google photos|gphotos|photos|google photo|google pics|google images':
        'com.google.android.apps.photos/com.google.android.apps.photos.home.HomeActivity',

    r'google calendar|gcal':
        'com.google.android.calendar/com.android.calendar.AllInOneActivity',

    r'camera':
        'com.android.camera2/com.android.camera.CameraLauncher',

    r'audio recorder':
        'com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.welcome.WelcomeActivity',

    r'google drive|gdrive|drive':
        'com.google.android.apps.docs/.drive.startup.StartupActivity',

    r'google keep|gkeep|keep':
        'com.google.android.keep/.activities.BrowseActivity',

    r'grubhub':
        'com.grubhub.android/com.grubhub.dinerapp.android.splash.SplashActivity',

    r'tripadvisor':
        'com.tripadvisor.tripadvisor/com.tripadvisor.android.ui.launcher.LauncherActivity',

    r'starbucks':
        'com.starbucks.mobilecard/.main.activity.LandingPageActivity',

    r'google docs|gdocs|docs':
        'com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',

    r'google sheets|gsheets|sheets':
        'com.google.android.apps.docs.editors.sheets/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',

    r'google slides|gslides|slides':
        'com.google.android.apps.docs.editors.slides/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',

    r'google voice|voice':
        'com.google.android.apps.googlevoice/com.google.android.apps.googlevoice.SplashActivity',

    r'clock':
        'com.google.android.deskclock/com.android.deskclock.DeskClock',

    r'google search|google':
        'com.google.android.googlequicksearchbox/com.google.android.googlequicksearchbox.SearchActivity',

    r'contacts':
        'com.google.android.contacts/com.android.contacts.activities.PeopleActivity',

    r'facebook|fb':
        'com.facebook.katana/com.facebook.katana.LoginActivity',

    r'whatsapp|wa':
        'com.whatsapp/com.whatsapp.Main',

    r'instagram|ig':
        'com.instagram.android/com.instagram.mainactivity.MainActivity',

    r'twitter|tweet':
        'com.twitter.android/com.twitter.app.main.MainActivity',

    r'snapchat|sc':
        'com.snapchat.android/com.snap.mushroom.MainActivity',

    r'telegram|tg':
        'org.telegram.messenger/org.telegram.ui.LaunchActivity',

    r'linkedin':
        'com.linkedin.android/com.linkedin.android.authenticator.LaunchActivity',

    r'spotify|spot':
        'com.spotify.music/com.spotify.music.MainActivity',

    r'netflix':
        'com.netflix.mediaclient/com.netflix.mediaclient.ui.launch.UIWebViewActivity',

    r'amazon shopping|amazon|amzn':
        'com.amazon.mShop.android.shopping/com.amazon.mShop.home.HomeActivity',

    r'tiktok|tt':
        'com.zhiliaoapp.musically/com.ss.android.ugc.aweme.splash.SplashActivity',

    r'discord':
        'com.discord/com.discord.app.AppActivity$Main',

    r'reddit':
        'com.reddit.frontpage/com.reddit.frontpage.MainActivity',

    r'pinterest':
        'com.pinterest/com.pinterest.activity.PinterestActivity',

    r'android world':
        'com.example.androidworld/.MainActivity',

    r'files':
        'com.google.android.documentsui/com.android.documentsui.files.FilesActivity',

    r'markor':
        'net.gsantner.markor/.activity.MainActivity',

    r'clipper':
        'ca.zgrs.clipper/ca.zgrs.clipper.Main',

    r'messages':
        'com.google.android.apps.messaging/com.google.android.apps.messaging.ui.ConversationListActivity',

    r'simple sms messenger|simple sms':
        'com.simplemobiletools.smsmessenger/com.simplemobiletools.smsmessenger.activities.MainActivity',

    r'dialer|phone':
        'com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity',

    r'simple calendar pro|simple calendar':
        'com.simplemobiletools.calendar.pro/com.simplemobiletools.calendar.pro.activities.MainActivity',

    r'simple gallery pro|simple gallery':
        'com.simplemobiletools.gallery.pro/com.simplemobiletools.gallery.pro.activities.MainActivity',

    r'miniwob':
        'com.google.androidenv.miniwob/com.google.androidenv.miniwob.app.MainActivity',

    r'simple draw pro':
        'com.simplemobiletools.draw.pro/com.simplemobiletools.draw.pro.activities.MainActivity',

    r'pro expense|pro expense app':
        'com.arduia.expense/com.arduia.expense.ui.MainActivity',

    r'broccoli|broccoli app|recipe app':
        'com.flauschcode.broccoli/com.flauschcode.broccoli.MainActivity',

    r'osmand':
        'net.osmand/net.osmand.plus.activities.MapActivity',

    r'tasks|tasks app':
        'org.tasks/com.todoroo.astrid.activity.MainActivity',

    r'opentracks|activity tracker':
        'de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity',

    r'joplin':
        'net.cozic.joplin/.MainActivity',

    r'vlc':
        'org.videolan.vlc/.gui.MainActivity',

    r'retro music|retro':
        'code.name.monkey.retromusic/.activities.MainActivity',
}


def get_screenshot(args, screenshot_path, scale=1.0, image_id=1):
    # 1️⃣ 设备截图（原分辨率）
    image_root = "/sdcard/screenshot" + str(image_id)
    command = args.adb_path + f" shell screencap -p {image_root}.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)

    # 2️⃣ 拉到本地
    if not args.on_device:
        command = args.adb_path + f" pull {image_root}.png {screenshot_path}"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    # 3️⃣ 本地 resize + 压缩
    time.sleep(0.3)
    img = Image.open(screenshot_path).convert("RGB")
    time.sleep(0.3)

    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w/scale), int(h/scale)), Image.BILINEAR)

    # 直接覆盖保存为 JPEG（减少 token）
    img.save(screenshot_path, format="JPEG", quality=85)


def get_a11y_tree(args, xml_path):
    command = args.adb_path + " shell uiautomator dump /sdcard/a11y.xml"
    subprocess.run(command, capture_output=True, text=True, shell=True)

    if not args.on_device:
        command = args.adb_path + f" pull /sdcard/a11y.xml {xml_path}"
        subprocess.run(command, capture_output=True, text=True, shell=True)
    #
    # # 设置路径
    # xml_path = "./screenshot/a11y.xml"
    #
    # # 检查是否成功
    # if not os.path.exists(xml_path):
    #     raise FileNotFoundError("Failed to retrieve window_dump.xml from device.")
    #
    # print(f"✅ Accessibility tree saved to {xml_path}")
    # return xml_path


def start_recording(adb_path):
    print("Remove existing screenrecord.mp4")
    command = adb_path + " shell rm /sdcard/screenrecord.mp4"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    print("Start!")
    # Use subprocess.Popen to allow terminating the recording process later
    command = adb_path + " shell screenrecord /sdcard/screenrecord.mp4"
    process = subprocess.Popen(command, shell=True)
    return process


def end_recording(adb_path, output_recording_path):
    print("Stopping recording...")
    # Send SIGINT to stop the screenrecord process gracefully
    stop_command = adb_path + " shell pkill -SIGINT screenrecord"
    subprocess.run(stop_command, capture_output=True, text=True, shell=True)
    sleep(1)  # Allow some time to ensure the recording is stopped
    
    print("Pulling recorded file from device...")
    pull_command = f"{adb_path} pull /sdcard/screenrecord.mp4 {output_recording_path}"
    subprocess.run(pull_command, capture_output=True, text=True, shell=True)
    print(f"Recording saved to {output_recording_path}")


def save_screenshot_to_file(adb_path, file_path="screenshot.png"):
    """
    Captures a screenshot from an Android device using ADB, saves it locally, and removes the screenshot from the device.

    Args:
        adb_path (str): The path to the adb executable.

    Returns:
        str: The path to the saved screenshot, or raises an exception on failure.
    """
    # Define the local filename for the screenshot
    local_file = file_path
    
    if os.path.dirname(local_file) != "":
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

    # Define the temporary file path on the Android device
    device_file = "/sdcard/screenshot.png"
    
    try:
        # print("\tRemoving existing screenshot from the Android device...")
        command = adb_path + " shell rm /sdcard/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)

        # Capture the screenshot on the device
        # print("\tCapturing screenshot on the Android device...")
        result = subprocess.run(f"{adb_path} shell screencap -p {device_file}", capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(f"Error: Failed to capture screenshot on the device. {result.stderr}")
        
        # Pull the screenshot to the local computer
        # print("\tTransferring screenshot to local computer...")
        result = subprocess.run(f"{adb_path} pull {device_file} {local_file}", capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(f"Error: Failed to transfer screenshot to local computer. {result.stderr}")
        
        # Remove the screenshot from the device
        # print("\tRemoving screenshot from the Android device...")
        result = subprocess.run(f"{adb_path} shell rm {device_file}", capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error: Failed to remove screenshot from the device. {result.stderr}")
        
        print(f"\tAtomic Operation Screenshot saved to {local_file}")
        return local_file
    
    except Exception as e:
        print(str(e))
        return None


def tap(adb_path, x, y):
    command = adb_path + f" shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type(adb_path, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == ' ':
            command = adb_path + f" shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == '_':
            command = adb_path + f" shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
            command = adb_path + f" shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in '-.,!?@\'°/:;()':
            command = adb_path + f" shell input text \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)

def enter(adb_path):
    command = adb_path + f" shell input keyevent KEYCODE_ENTER"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def swipe(adb_path, x1, y1, x2, y2):
    command = adb_path + f" shell input swipe {x1} {y1} {x2} {y2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def back(adb_path):
    command = adb_path + f" shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    
    
def home(adb_path):
    # command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    command = adb_path + f" shell input keyevent KEYCODE_HOME"
    subprocess.run(command, capture_output=True, text=True, shell=True)

def switch_app(adb_path):
    command = adb_path + f" shell input keyevent KEYCODE_APP_SWITCH"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def launch_app(adb: str, app_name: str):
    package = normalize_app_name(app_name)
    print(f"[ADB] launcher start → {package}")
    os.system(f"{adb} shell monkey -p {package} -c android.intent.category.LAUNCHER 1")
    time.sleep(0.1)
    return package


def normalize_app_name(app_name: str) -> str:
    """Map human-readable app names to Android package names."""
    APP_NAME_TO_PACKAGE = {
        # ===== File / Media =====
        "File Manager": "com.google.android.documentsui",
        "Files": "com.simplemobiletools.filemanager.pro",
        "Gallery": "com.simplemobiletools.gallery.pro",
        "Photos": "com.google.android.apps.photos",

        # ===== Audio / Video =====
        "Audio Recorder": "com.dimowner.audiorecorder",  # 注意：不是 dimowner
        "Music": "com.spotify.music",
        "YouTube Music": "com.google.android.apps.youtube.music",
        "YouTube": "com.google.android.youtube",

        # ===== Notes / Docs =====
        "Notes": "com.simplemobiletools.notes.pro",
        "Docs": "com.google.android.apps.docs",

        # ===== Communication =====
        "Phone": "com.google.android.dialer",
        "Messages": "com.google.android.apps.messaging",
        "Contacts": "com.google.android.contacts",
        "Gmail": "com.google.android.gm",

        # ===== Browser / Search =====
        "Chrome": "com.android.chrome",
        "Browser": "com.android.chrome",
        "Google": "com.google.android.googlequicksearchbox",

        # ===== Utilities =====
        "Calculator": "com.simplemobiletools.calculator",
        "Clock": "com.simplemobiletools.clock",
        "Calendar": "com.simplemobiletools.calendar",
        "Flashlight": "com.simplemobiletools.flashlight",

        # ===== Maps =====
        "Maps": "com.google.android.apps.maps",

        # ===== Expense / Tasks (Android World 常用) =====
        "Tasks": "org.tasks",
        "Pro Expense": "com.arduia.expense",

        # ===== Launcher（一般不需要 monkey）=====
        "Launcher": "com.google.android.apps.nexuslauncher",
    }
    return APP_NAME_TO_PACKAGE.get(app_name, app_name)