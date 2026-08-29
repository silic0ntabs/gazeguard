-- GazeGuard Blur Policy Engine for mpv / IINA
-- Reads Show.S01E01.halal.json (descriptive SemanticMeta), applies a screen action
-- (blur_mild / blur_strong / none) per interval based on the ACTIVE_PROFILE.
-- Audio + subs intact (subtitles composite ON TOP of the filtered frame).
--
-- NOTE: decide_screen_action(meta) MUST mirror gazeguard/policy.action_for() 1:1.

local utils = require 'mp.utils'
local msg = require 'mp.msg'

local intervals = {}          -- { start, end, meta } sorted by start
local is_active = false       -- some filter is currently applied
local manual_override = false

-- Profile. Change here or via keybind (cycled with Ctrl+B).
local ACTIVE_PROFILE = "STRICT"
local PROFILES = { "STRICT", "BALANCED", "MINIMAL" }

local label = "gazeguard_blur"
local blurLabel = "gazeguard_strong"

--------------------------------------------------------------------------------
-- Load sidecar (Show.Ep.halal.json preferred, fall back to Show.Ep.json)
--------------------------------------------------------------------------------
local function file_ok(p)
    local info = utils.file_info(p)
    return info and info.is_file
end

-- Build a robust path to the sidecar regardless of how mpv resolved `path`
-- (relative vs absolute). Collect candidate BASE prefixes (path with extension stripped),
-- then try "<base>.halal.json" / "<base>.json" for each.
local function find_sidecar(path)
    if not path then return nil end
    local wd = mp.get_property("working-directory") or "."
    local bases = {}
    local push_base = function(p)
        if p and p ~= "" then
            p = p:gsub("%?.*$", "")          -- strip ?query / ?title=
            p = p:gsub("%.%w+$", "")         -- strip one extension
            if p ~= "" then bases[#bases+1] = p end
        end
    end
    -- 1. path as-is (absolute, or relative -> resolved by file_info against mpv CWD)
    push_base(path)
    -- 2. path joined with working-directory (handler for relative paths that file_info
    --    couldn't resolve because mpv chdir'd elsewhere)
    if not path:find("^[/\\]") and not path:find("^%a+:") then
        push_base(wd .. "/" .. path)
    end
    -- 3. directory of the video (from mpv's file-local-path) + video basename
    local local_file = mp.get_property("file-local-path") or path
    local lfdir = local_file:match("^(.*)[/\\][^/\\]+$")
    local lfbase = local_file:match("([^/\\]+)$")
    if lfdir and lfbase then
        push_base(lfdir .. "/" .. lfbase)
    end

    local seen = {}
    for _, base in ipairs(bases) do
        if not seen[base] then
            seen[base] = true
            local j = base .. ".halal.json"
            if file_ok(j) then return j end
            j = base .. ".json"
            if file_ok(j) then return j end
        end
    end
    return nil
end

function load_intervals()
    local path = mp.get_property("path") or mp.get_property("file-local-path")
    if not path then msg.warn("No video path"); return end
    local json_path = find_sidecar(path)
    if not json_path then
        msg.info("No GazeGuard sidecar for: " .. tostring(mp.get_property("filename")))
        intervals = {}
        return
    end
    local f = io.open(json_path, "r")
    if not f then intervals = {}; return end
    local content = f:read("*all"); f:close()
    local data = utils.parse_json(content)
    -- Accept either a bare list, or {"intervals":[...], ...}
    intervals = (type(data) == "table" and data.intervals) or (type(data) == "table" and #data > 0 and data) or {}
    if type(intervals) ~= "table" then intervals = {} end
    table.sort(intervals, function(a, b) return (a.start or 0) < (b.start or 0) end)
    msg.info("Loaded " .. #intervals .. " GazeGuard interval(s) from " .. json_path .. " profile=" .. ACTIVE_PROFILE)
end

--------------------------------------------------------------------------------
-- Policy engine (mirror of policy.action_for)
--------------------------------------------------------------------------------
function hard_trigger(meta)
    if not meta then return false end
    return meta.exposure_level == "explicit"
        or meta.intimacy_level == "explicit"
        or (meta.intimacy_level == "sensual_romance"
            and (meta.exposure_level == "moderate" or meta.exposure_level == "explicit"))
end

function clinical_case(meta)
    return meta.situational_state == "clinical_or_distress"
        and meta.physical_action == "platonic_or_combat"
        and meta.exposure_level ~= "explicit"
end

function decide_screen_action(meta)
    if not meta then return "none" end
    if hard_trigger(meta) then
        -- confirmed explicit -> STRONG blur (mirror of Action.BLUR_STRONG)
        return "blur_strong"
    end

    local clinical = clinical_case(meta)

    if ACTIVE_PROFILE == "STRICT" then
        -- mirror of policy.py: intimacy/implied_intimacy only blur when backed by real
        -- exposed skin (moderate/explicit). A couple talking in bed or a car with nothing
        -- exposed passes. Real content (lingerie/bra) is moderate/explicit and still blurs.
        local skin = meta.exposure_level == "moderate" or meta.exposure_level == "explicit"
        local objectifying = meta.camera_intent == "voyeuristic_objectifying"
            or meta.visual_layer == "background_ambient"
        local explicit_ambient = meta.situational_state == "ambient_entertainment"
            and meta.exposure_level == "explicit"
        if not clinical and (
            ((meta.intimacy_level == "mild_romance" or meta.intimacy_level == "sensual_romance") and skin)
            or (meta.situational_state == "implied_intimacy" and skin)
            or explicit_ambient
            or (meta.camera_intent == "voyeuristic_objectifying"
                and (meta.exposure_level == "mild" or meta.exposure_level == "moderate"))) then
            return "blur_strong"
        end
        if objectifying and (meta.exposure_level == "mild" or meta.exposure_level == "moderate") then
            return "blur_strong"
        end
        if clinical and meta.exposure_level == "moderate" then
            return "blur_mild"
        end

    elseif ACTIVE_PROFILE == "BALANCED" then
        local skin = meta.exposure_level == "moderate" or meta.exposure_level == "explicit"
        if not clinical and (
            (meta.intimacy_level == "sensual_romance" and skin)
            or (meta.situational_state == "implied_intimacy" and skin)) then
            return "blur_strong"
        end
        if meta.exposure_level == "moderate"
            or (meta.intimacy_level == "mild_romance" and skin)
            or meta.camera_intent == "voyeuristic_objectifying"
            or meta.visual_layer == "background_ambient" then
            return "blur_mild"
        end

    elseif ACTIVE_PROFILE == "MINIMAL" then
        -- hard triggers only (actual explicit acts/nudity); tame kisses and implied romance pass
    end

    return "none"
end

--------------------------------------------------------------------------------
-- Apply / clear filters
--------------------------------------------------------------------------------
local function clear_filters()
    if is_active then
        mp.commandv("vf", "remove", "@" .. label)
        mp.commandv("vf", "remove", "@" .. blurLabel)
        is_active = false
    end
end

local function apply_filters(action)
    clear_filters()
    if action == "blur_strong" then
        -- confirmed explicit: heavy gaussian + a touch of darkening so content is obscured,
        -- while mpv's subtitle renderer (composited on top) keeps dialogue readable.
        mp.commandv("vf", "add", "@" .. blurLabel .. ":gblur=sigma=60")
        mp.commandv("vf", "add", "@" .. label .. ":eq=brightness=-0.15")
        is_active = true
    elseif action == "blur_mild" then
        -- suggestive-but-uncertain: frosted blur, scene still perceivable
        mp.commandv("vf", "add", "@" .. blurLabel .. ":gblur=sigma=18")
        is_active = true
    end
end

--------------------------------------------------------------------------------
-- Watch playback clock
--------------------------------------------------------------------------------
function check_time(name, value)
    if #intervals == 0 or manual_override then return end
    if not value then return end
    for _, iv in ipairs(intervals) do
        if value >= (iv.start or 0) and value <= (iv["end"] or 0) then
            apply_filters(decide_screen_action(iv.meta or {}))
            return
        end
    end
    clear_filters()
end

-- Manual unlock (Shift+B)
function toggle_manual()
    manual_override = not manual_override
    if manual_override then
        mp.osd_message("GazeGuard: Manual Unlock", 2)
        clear_filters()
    else
        mp.osd_message("GazeGuard: Auto mode restored", 2)
        check_time(nil, mp.get_property_number("time-pos"))
    end
end

-- Cycle profiles (Ctrl+B) — no API calls, just a playback decision change
function cycle_profile()
    for i, p in ipairs(PROFILES) do
        if ACTIVE_PROFILE == p then
            ACTIVE_PROFILE = PROFILES[i % #PROFILES + 1]
            break
        end
    end
    mp.osd_message("GazeGuard profile: " .. ACTIVE_PROFILE, 2)
    check_time(nil, mp.get_property_number("time-pos"))
end

--------------------------------------------------------------------------------
-- Init
--------------------------------------------------------------------------------
mp.register_event("start-file", function()
    intervals = {}
    is_active = false
    manual_override = false
    load_intervals()
end)

-- Playback polish: guarantee subtitle rendering stays ON during video drops (blackouts
-- keep audio+subs so dialogue comprehension is never lost). We set it once here rather than
-- toggling, since our filter approach (drawbox/sub composition) already keeps subs visible.
mp.observe_property("sub-visibility", "string", function()
    if mp.get_property("sub-visibility") == "no" then
        msg.info("Subtitle rendering disabled by user — keeping it OFF.")
    end
end)

mp.observe_property("time-pos", "number", check_time)
mp.add_key_binding("Shift+B", "gazeguard-toggle", toggle_manual)
mp.add_key_binding("Ctrl+B", "gazeguard-profile", cycle_profile)

msg.info("GazeGuard policy engine loaded. Hotkeys: Shift+B unlock, Ctrl+B profile (" .. ACTIVE_PROFILE .. ")")