import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
  useMemo,
} from "react";
import { useStorage } from "./StorageContext";
import { StorageKeys } from "./StorageContext";

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

/** Relative luminance (0–1). Above ~0.45 we use black text for contrast. */
function getLuminance(hex: string): number {
  const rgb = hexToRgb(hex);
  if (!rgb) return 0;
  const { r, g, b } = rgb;
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

const DEFAULT_BLOCK_COLOR = "#212529";

type BlockColorContextValue = {
  blockColor: string;
  setBlockColor: (color: string) => void;
  isLight: boolean;
};

const BlockColorContext = createContext<BlockColorContextValue | null>(null);

function BlockColorProvider({ children }: { children: ReactNode }) {
  const storage = useStorage();
  const [blockColor, setBlockColorState] = useState<string>(() => {
    const saved = storage.getLocalStorage(StorageKeys.BLOCK_COLOR);
    return saved || DEFAULT_BLOCK_COLOR;
  });

  useEffect(() => {
    const saved = storage.getLocalStorage(StorageKeys.BLOCK_COLOR);
    if (saved) setBlockColorState(saved);
  }, []);

  const setBlockColor = (color: string) => {
    setBlockColorState(color);
    storage.setLocalStorage(StorageKeys.BLOCK_COLOR, color);
  };

  const isLight = useMemo(
    () => getLuminance(blockColor) > 0.45,
    [blockColor]
  );

  const value = useMemo<BlockColorContextValue>(
    () => ({ blockColor, setBlockColor, isLight }),
    [blockColor, isLight]
  );

  return (
    <BlockColorContext.Provider value={value}>
      {children}
    </BlockColorContext.Provider>
  );
}

function useBlockColor(): BlockColorContextValue {
  const ctx = useContext(BlockColorContext);
  if (!ctx) {
    throw new Error("useBlockColor must be used within BlockColorProvider");
  }
  return ctx;
}

export { BlockColorProvider, useBlockColor, DEFAULT_BLOCK_COLOR, getLuminance };
