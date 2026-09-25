/** One pixel-art image: each row character indexes `palette`; "." is transparent. */
export type Art = {
  palette: Record<string, string>;
  rows: string[];
};
